import Mathlib

namespace NUMINAMATH_CALUDE_fraction_of_single_men_l3476_347656

theorem fraction_of_single_men
  (total : ℕ)
  (h_total_pos : total > 0)
  (women_ratio : ℚ)
  (h_women_ratio : women_ratio = 70 / 100)
  (married_ratio : ℚ)
  (h_married_ratio : married_ratio = 40 / 100)
  (married_men_ratio : ℚ)
  (h_married_men_ratio : married_men_ratio = 2 / 3)
  : (total - women_ratio * total - married_men_ratio * (total - women_ratio * total)) / (total - women_ratio * total) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_single_men_l3476_347656


namespace NUMINAMATH_CALUDE_boys_passed_exam_l3476_347694

/-- Proves the number of boys who passed an examination given specific conditions -/
theorem boys_passed_exam (total_boys : ℕ) (overall_avg : ℚ) (pass_avg : ℚ) (fail_avg : ℚ) :
  total_boys = 120 →
  overall_avg = 36 →
  pass_avg = 39 →
  fail_avg = 15 →
  ∃ (passed_boys : ℕ),
    passed_boys = 105 ∧
    passed_boys ≤ total_boys ∧
    (passed_boys : ℚ) * pass_avg + (total_boys - passed_boys : ℚ) * fail_avg = (total_boys : ℚ) * overall_avg :=
by sorry

end NUMINAMATH_CALUDE_boys_passed_exam_l3476_347694


namespace NUMINAMATH_CALUDE_discount_calculation_l3476_347673

-- Define the initial discount
def initial_discount : ℝ := 0.40

-- Define the additional discount
def additional_discount : ℝ := 0.10

-- Define the claimed total discount
def claimed_discount : ℝ := 0.55

-- Theorem to prove the actual discount and the difference
theorem discount_calculation :
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_additional
  let discount_difference := claimed_discount - actual_discount
  actual_discount = 0.46 ∧ discount_difference = 0.09 := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l3476_347673


namespace NUMINAMATH_CALUDE_train_length_l3476_347670

/-- Given a train with constant speed that crosses a tree in 120 seconds
    and passes a 700m long platform in 190 seconds,
    the length of the train is 1200 meters. -/
theorem train_length (speed : ℝ) (train_length : ℝ) : 
  (train_length / 120 = speed) →
  ((train_length + 700) / 190 = speed) →
  train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3476_347670


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3476_347669

theorem rectangular_prism_volume (a b c : ℕ) : 
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 →
  2 * ((a - 2) * (b - 2) + (b - 2) * (c - 2) + (a - 2) * (c - 2)) = 24 →
  4 * ((a - 2) + (b - 2) + (c - 2)) = 28 →
  a * b * c = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3476_347669


namespace NUMINAMATH_CALUDE_parabola_intersection_l3476_347619

-- Define the two parabola functions
def f (x : ℝ) : ℝ := 3 * x^2 - 15 * x - 4
def g (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 8

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(3, -22), (4, -16)}

-- Theorem statement
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) ∈ intersection_points :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3476_347619


namespace NUMINAMATH_CALUDE_store_profit_optimization_l3476_347608

/-- Represents the store's sales and profit model -/
structure StoreSalesModel where
  purchase_price : ℕ
  initial_selling_price : ℕ
  initial_monthly_sales : ℕ
  additional_sales_per_yuan : ℕ

/-- Calculates the monthly profit given a price reduction -/
def monthly_profit (model : StoreSalesModel) (price_reduction : ℕ) : ℕ :=
  let new_price := model.initial_selling_price - price_reduction
  let new_sales := model.initial_monthly_sales + model.additional_sales_per_yuan * price_reduction
  (new_price - model.purchase_price) * new_sales

/-- Theorem stating the initial monthly profit and the optimal price reduction -/
theorem store_profit_optimization (model : StoreSalesModel) 
  (h1 : model.purchase_price = 280)
  (h2 : model.initial_selling_price = 360)
  (h3 : model.initial_monthly_sales = 60)
  (h4 : model.additional_sales_per_yuan = 5) :
  (monthly_profit model 0 = 4800) ∧ 
  (monthly_profit model 60 = 7200) ∧ 
  (∀ x, x ≠ 60 → monthly_profit model x ≤ 7200) :=
sorry

end NUMINAMATH_CALUDE_store_profit_optimization_l3476_347608


namespace NUMINAMATH_CALUDE_min_value_of_f_l3476_347639

/-- The base-10 logarithm function -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The function to be minimized -/
noncomputable def f (x : ℝ) : ℝ := lg x + (Real.log 10) / (Real.log x)

theorem min_value_of_f :
  ∀ x > 1, f x ≥ 2 ∧ f 10 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3476_347639


namespace NUMINAMATH_CALUDE_sum_of_squares_bounds_l3476_347629

/-- A quadrilateral inscribed in a unit square -/
structure InscribedQuadrilateral where
  w : Real
  x : Real
  y : Real
  z : Real
  w_in_range : 0 ≤ w ∧ w ≤ 1
  x_in_range : 0 ≤ x ∧ x ≤ 1
  y_in_range : 0 ≤ y ∧ y ≤ 1
  z_in_range : 0 ≤ z ∧ z ≤ 1

/-- The sum of squares of the sides of an inscribed quadrilateral -/
def sumOfSquares (q : InscribedQuadrilateral) : Real :=
  (q.w^2 + q.x^2) + ((1-q.x)^2 + q.y^2) + ((1-q.y)^2 + q.z^2) + ((1-q.z)^2 + (1-q.w)^2)

/-- Theorem: The sum of squares of the sides of a quadrilateral inscribed in a unit square is between 2 and 4 -/
theorem sum_of_squares_bounds (q : InscribedQuadrilateral) : 
  2 ≤ sumOfSquares q ∧ sumOfSquares q ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_bounds_l3476_347629


namespace NUMINAMATH_CALUDE_set_in_proportion_l3476_347618

/-- A set of four numbers (a, b, c, d) is in proportion if a:b = c:d -/
def IsInProportion (a b c d : ℚ) : Prop :=
  a * d = b * c

/-- Prove that the set (1, 2, 2, 4) is in proportion -/
theorem set_in_proportion :
  IsInProportion 1 2 2 4 := by
  sorry

end NUMINAMATH_CALUDE_set_in_proportion_l3476_347618


namespace NUMINAMATH_CALUDE_initial_average_runs_l3476_347607

theorem initial_average_runs (initial_matches : ℕ) (additional_runs : ℕ) (average_increase : ℕ) : 
  initial_matches = 10 →
  additional_runs = 89 →
  average_increase = 5 →
  ∃ (initial_average : ℕ),
    (initial_matches * initial_average + additional_runs) / (initial_matches + 1) = initial_average + average_increase ∧
    initial_average = 34 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_runs_l3476_347607


namespace NUMINAMATH_CALUDE_saving_fraction_is_one_fourth_l3476_347640

/-- Represents the worker's monthly savings behavior -/
structure WorkerSavings where
  monthlyPay : ℝ
  savingFraction : ℝ
  monthlyPay_pos : 0 < monthlyPay
  savingFraction_range : 0 ≤ savingFraction ∧ savingFraction ≤ 1

/-- The theorem stating that the saving fraction is 1/4 given the conditions -/
theorem saving_fraction_is_one_fourth (w : WorkerSavings) 
  (h : 12 * w.savingFraction * w.monthlyPay = 
       4 * (1 - w.savingFraction) * w.monthlyPay) : 
  w.savingFraction = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_saving_fraction_is_one_fourth_l3476_347640


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_range_of_a_for_not_r_necessary_not_sufficient_for_not_p_l3476_347605

-- Define the predicates p, q, and r
def p (x : ℝ) : Prop := |3*x - 4| > 2
def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0
def r (x a : ℝ) : Prop := (x - a) * (x - a - 1) < 0

-- Define the negations of p, q, and r
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)
def not_r (x a : ℝ) : Prop := ¬(r x a)

-- Theorem 1: ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ ¬(∀ x, not_q x → not_p x) :=
sorry

-- Theorem 2: Range of a for which ¬r is a necessary but not sufficient condition for ¬p
theorem range_of_a_for_not_r_necessary_not_sufficient_for_not_p :
  ∀ a, (∀ x, not_p x → not_r x a) ∧ ¬(∀ x, not_r x a → not_p x) ↔ (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_range_of_a_for_not_r_necessary_not_sufficient_for_not_p_l3476_347605


namespace NUMINAMATH_CALUDE_special_multiples_count_l3476_347632

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_multiples (n : ℕ) : ℕ :=
  count_multiples n 5 + count_multiples n 6 - count_multiples n 15

theorem special_multiples_count :
  count_special_multiples 3000 = 900 := by sorry

end NUMINAMATH_CALUDE_special_multiples_count_l3476_347632


namespace NUMINAMATH_CALUDE_joan_has_three_marbles_l3476_347610

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := 12

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := total_marbles - mary_marbles

theorem joan_has_three_marbles : joan_marbles = 3 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_three_marbles_l3476_347610


namespace NUMINAMATH_CALUDE_center_value_is_35_l3476_347602

/-- Represents a 4x4 array where each row and column forms an arithmetic sequence -/
def ArithmeticArray := Matrix (Fin 4) (Fin 4) ℝ

/-- Checks if a row or column is an arithmetic sequence -/
def is_arithmetic_sequence (seq : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i j : Fin 4, i.val < j.val → seq j - seq i = d * (j.val - i.val)

/-- Definition of our specific arithmetic array -/
def special_array (A : ArithmeticArray) : Prop :=
  (∀ i : Fin 4, is_arithmetic_sequence (λ j => A i j)) ∧ 
  (∀ j : Fin 4, is_arithmetic_sequence (λ i => A i j)) ∧
  A 0 0 = 3 ∧ A 0 3 = 27 ∧ A 3 0 = 6 ∧ A 3 3 = 66

/-- The center value of the array -/
def center_value (A : ArithmeticArray) : ℝ := A 1 1

theorem center_value_is_35 (A : ArithmeticArray) (h : special_array A) : 
  center_value A = 35 := by
  sorry

end NUMINAMATH_CALUDE_center_value_is_35_l3476_347602


namespace NUMINAMATH_CALUDE_conference_attendance_l3476_347620

/-- The number of writers at the conference -/
def writers : ℕ := 45

/-- The number of editors at the conference -/
def editors : ℕ := 37

/-- The number of people who are both writers and editors -/
def both : ℕ := 18

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * both

/-- The total number of people attending the conference -/
def total : ℕ := writers + editors - both + neither

theorem conference_attendance :
  editors > 36 ∧ both ≤ 18 → total = 100 := by sorry

end NUMINAMATH_CALUDE_conference_attendance_l3476_347620


namespace NUMINAMATH_CALUDE_solve_equation_l3476_347614

theorem solve_equation (b : ℚ) (h : b + b / 4 = 10 / 4) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3476_347614


namespace NUMINAMATH_CALUDE_total_cost_for_nuggets_l3476_347612

-- Define the number of chicken nuggets ordered
def total_nuggets : ℕ := 100

-- Define the number of nuggets in a box
def nuggets_per_box : ℕ := 20

-- Define the cost of one box
def cost_per_box : ℕ := 4

-- Theorem to prove
theorem total_cost_for_nuggets : 
  (total_nuggets / nuggets_per_box) * cost_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_for_nuggets_l3476_347612


namespace NUMINAMATH_CALUDE_petya_running_time_l3476_347677

theorem petya_running_time (V D : ℝ) (hV : V > 0) (hD : D > 0) : 
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := D / (2 * V1)
  let T2 := D / (2 * V2)
  let Tactual := T1 + T2
  Tactual > T :=
by sorry

end NUMINAMATH_CALUDE_petya_running_time_l3476_347677


namespace NUMINAMATH_CALUDE_perfect_square_sum_l3476_347653

theorem perfect_square_sum (n : ℕ) : 
  n > 0 ∧ n < 200 ∧ (∃ k : ℕ, n^2 + (n+1)^2 = k^2) ↔ n = 3 ∨ n = 20 ∨ n = 119 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l3476_347653


namespace NUMINAMATH_CALUDE_masons_father_age_l3476_347648

/-- Given the ages of Mason and Sydney, and their relationship to Mason's father's age,
    prove that Mason's father is 66 years old. -/
theorem masons_father_age (mason_age sydney_age father_age : ℕ) : 
  mason_age = 20 →
  sydney_age = 3 * mason_age →
  father_age = sydney_age + 6 →
  father_age = 66 := by
  sorry

end NUMINAMATH_CALUDE_masons_father_age_l3476_347648


namespace NUMINAMATH_CALUDE_divide_multiply_problem_l3476_347691

theorem divide_multiply_problem : (2.25 / 3) * 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_divide_multiply_problem_l3476_347691


namespace NUMINAMATH_CALUDE_problem_statement_l3476_347681

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 2) :
  (1 < b ∧ b < 2) ∧ a * b < 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3476_347681


namespace NUMINAMATH_CALUDE_deschamps_farm_l3476_347675

theorem deschamps_farm (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 160) 
  (h2 : total_legs = 400) : ∃ (chickens cows : ℕ),
  chickens + cows = total_animals ∧ 
  2 * chickens + 4 * cows = total_legs ∧ 
  cows = 40 := by
  sorry

end NUMINAMATH_CALUDE_deschamps_farm_l3476_347675


namespace NUMINAMATH_CALUDE_injective_function_equality_l3476_347668

theorem injective_function_equality (f : ℕ → ℕ) (h_inj : Function.Injective f) 
  (h_cond : ∀ n : ℕ, f (f n) ≤ (f n + n) / 2) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_injective_function_equality_l3476_347668


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l3476_347615

/-- Represents the number of valid arrangements for n coins where no three consecutive coins are face to face to face -/
def validArrangements : Nat → Nat
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => validArrangements (n + 2) + validArrangements (n + 1) + validArrangements n

/-- The number of ways to choose 5 positions out of 10 for gold coins -/
def colorDistributions : Nat := Nat.choose 10 5

/-- The total number of distinguishable arrangements of 5 gold and 5 silver coins
    with the given face-to-face constraint -/
def totalArrangements : Nat := colorDistributions * validArrangements 10

theorem coin_stack_arrangements :
  totalArrangements = 69048 := by
  sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l3476_347615


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l3476_347661

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole1_diameter : ℝ := 4
  let hole2_diameter : ℝ := 4
  let hole3_diameter : ℝ := 3
  let hole_depth : ℝ := 6
  
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let hole1_volume := π * (hole1_diameter / 2) ^ 2 * hole_depth
  let hole2_volume := π * (hole2_diameter / 2) ^ 2 * hole_depth
  let hole3_volume := π * (hole3_diameter / 2) ^ 2 * hole_depth
  
  sphere_volume - (hole1_volume + hole2_volume + hole3_volume) = 2242.5 * π :=
by sorry


end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l3476_347661


namespace NUMINAMATH_CALUDE_angle_bisector_product_not_unique_l3476_347693

/-- A triangle represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The product of the lengths of the three angle bisectors of a triangle -/
def angle_bisector_product (t : Triangle) : ℝ := sorry

/-- Statement: The product of the three angle bisectors does not uniquely determine a triangle -/
theorem angle_bisector_product_not_unique :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ angle_bisector_product t1 = angle_bisector_product t2 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_product_not_unique_l3476_347693


namespace NUMINAMATH_CALUDE_friend_team_assignments_l3476_347686

/-- The number of ways to assign n distinguishable objects to k distinct categories -/
def assignments (n k : ℕ) : ℕ := k^n

/-- The number of friends -/
def num_friends : ℕ := 8

/-- The number of teams -/
def num_teams : ℕ := 4

/-- Theorem: The number of ways to assign 8 friends to 4 teams is 65536 -/
theorem friend_team_assignments : assignments num_friends num_teams = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignments_l3476_347686


namespace NUMINAMATH_CALUDE_perpendicular_similarity_l3476_347688

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop :=
  sorry -- Definition of acute triangle

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry -- Definition of point being inside a triangle

/-- Constructs a new triangle by dropping perpendiculars from a point to the sides of another triangle -/
def dropPerpendiculars (p : Point) (t : Triangle) : Triangle :=
  sorry -- Definition of dropping perpendiculars

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  sorry -- Definition of triangle similarity

theorem perpendicular_similarity 
  (ABC : Triangle) 
  (P : Point) 
  (h_acute : isAcute ABC) 
  (h_inside : isInside P ABC) : 
  let A₁B₁C₁ := dropPerpendiculars P ABC
  let A₂B₂C₂ := dropPerpendiculars P A₁B₁C₁
  let A₃B₃C₃ := dropPerpendiculars P A₂B₂C₂
  areSimilar A₃B₃C₃ ABC :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_similarity_l3476_347688


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3476_347622

/-- A regular polygon with side length 7 units and exterior angle 45 degrees has a perimeter of 56 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (h1 : s = 7) (h2 : θ = 45) :
  let n : ℝ := 360 / θ
  let perimeter : ℝ := n * s
  perimeter = 56 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3476_347622


namespace NUMINAMATH_CALUDE_problem_solution_l3476_347646

theorem problem_solution (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 64)
  (sum_prod : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3476_347646


namespace NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l3476_347636

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ 
    a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l3476_347636


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3476_347600

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 252) 
  (h2 : a*b + b*c + c*a = 116) : 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3476_347600


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_24_l3476_347601

theorem factorial_ratio_equals_24 :
  ∃! (n : ℕ), n > 3 ∧ n.factorial / (n - 3).factorial = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_24_l3476_347601


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l3476_347674

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0) ∧ (2 * 5^2 + 3 * 5 - k = 0) → k = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l3476_347674


namespace NUMINAMATH_CALUDE_range_of_a_for_two_zeros_l3476_347628

noncomputable def f (a x : ℝ) : ℝ := 
  if x ≥ a then x else x^3 - 3*x

noncomputable def g (a x : ℝ) : ℝ := 2 * f a x - a * x

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ g a x = 0 ∧ g a y = 0) ∧
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → ¬(g a x = 0 ∧ g a y = 0 ∧ g a z = 0)) →
  a > -3/2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_zeros_l3476_347628


namespace NUMINAMATH_CALUDE_tree_boy_growth_rate_ratio_l3476_347624

/-- Given the initial and final heights of a tree and a boy, calculate the ratio of their growth rates. -/
theorem tree_boy_growth_rate_ratio
  (tree_initial : ℝ) (tree_final : ℝ)
  (boy_initial : ℝ) (boy_final : ℝ)
  (h_tree_initial : tree_initial = 16)
  (h_tree_final : tree_final = 40)
  (h_boy_initial : boy_initial = 24)
  (h_boy_final : boy_final = 36) :
  (tree_final - tree_initial) / (boy_final - boy_initial) = 2 := by
sorry

end NUMINAMATH_CALUDE_tree_boy_growth_rate_ratio_l3476_347624


namespace NUMINAMATH_CALUDE_marbles_fraction_l3476_347682

theorem marbles_fraction (initial_marbles : ℕ) (fraction_taken : ℚ) (cleo_final : ℕ) : 
  initial_marbles = 30 →
  fraction_taken = 3/5 →
  cleo_final = 15 →
  (cleo_final - (fraction_taken * initial_marbles / 2)) / (initial_marbles - fraction_taken * initial_marbles) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_marbles_fraction_l3476_347682


namespace NUMINAMATH_CALUDE_result_calculation_l3476_347626

theorem result_calculation (h1 : 7125 / 1.25 = 5700) (h2 : x = 3) : 
  (712.5 / 12.5) ^ x = 185193 := by
sorry

end NUMINAMATH_CALUDE_result_calculation_l3476_347626


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l3476_347637

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 5 indistinguishable balls -/
def num_balls : ℕ := 5

/-- There are 3 distinguishable boxes -/
def num_boxes : ℕ := 3

/-- The theorem states that there are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : 
  distribute_balls num_balls num_boxes = 21 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l3476_347637


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3476_347692

theorem no_integer_solutions : ¬ ∃ (x : ℤ), x^2 - 9*x + 20 < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3476_347692


namespace NUMINAMATH_CALUDE_letters_per_large_envelope_l3476_347645

theorem letters_per_large_envelope 
  (total_letters : ℕ) 
  (small_envelope_letters : ℕ) 
  (large_envelopes : ℕ) 
  (h1 : total_letters = 80) 
  (h2 : small_envelope_letters = 20) 
  (h3 : large_envelopes = 30) : 
  (total_letters - small_envelope_letters) / large_envelopes = 2 := by
  sorry

end NUMINAMATH_CALUDE_letters_per_large_envelope_l3476_347645


namespace NUMINAMATH_CALUDE_parallelogram_height_l3476_347638

theorem parallelogram_height (area base height : ℝ) : 
  area = 480 ∧ base = 32 ∧ area = base * height → height = 15 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3476_347638


namespace NUMINAMATH_CALUDE_sum_of_digits_8_to_1002_l3476_347679

theorem sum_of_digits_8_to_1002 :
  let n := 8^1002
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_to_1002_l3476_347679


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3476_347662

/-- Represents the composition of a school population -/
structure SchoolPopulation where
  teachers : ℕ
  male_students : ℕ
  female_students : ℕ

/-- Represents a stratified sample from the school population -/
structure StratifiedSample where
  total_size : ℕ
  female_sample : ℕ

/-- Theorem: Given a school population and a stratified sample where 80 people are drawn
    from the female students, the total sample size is 192 -/
theorem stratified_sample_size 
  (pop : SchoolPopulation) 
  (sample : StratifiedSample) :
  pop.teachers = 200 →
  pop.male_students = 1200 →
  pop.female_students = 1000 →
  sample.female_sample = 80 →
  sample.total_size = 192 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l3476_347662


namespace NUMINAMATH_CALUDE_inequalities_proof_l3476_347634

theorem inequalities_proof (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 4) : 
  (1 / a + 1 / (b + 1) ≥ 4 / 5) ∧ 
  (4 / (a * b) + a / b ≥ (Real.sqrt 5 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3476_347634


namespace NUMINAMATH_CALUDE_robot_position_difference_l3476_347609

-- Define the robot's position function
def robot_position (n : ℕ) : ℤ :=
  let full_cycles := n / 7
  let remainder := n % 7
  let cycle_progress := if remainder ≤ 4 then remainder else 4 - (remainder - 4)
  full_cycles + cycle_progress

-- State the theorem
theorem robot_position_difference : robot_position 2007 - robot_position 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_robot_position_difference_l3476_347609


namespace NUMINAMATH_CALUDE_cafeteria_apples_l3476_347664

/-- The number of apples handed out to students -/
def apples_to_students : ℕ := 27

/-- The number of pies that can be made -/
def number_of_pies : ℕ := 5

/-- The number of apples needed for each pie -/
def apples_per_pie : ℕ := 4

/-- The total number of apples in the cafeteria initially -/
def total_apples : ℕ := apples_to_students + number_of_pies * apples_per_pie

theorem cafeteria_apples : total_apples = 47 := by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l3476_347664


namespace NUMINAMATH_CALUDE_BA_is_2I_l3476_347660

theorem BA_is_2I (A : Matrix (Fin 4) (Fin 2) ℝ) (B : Matrix (Fin 2) (Fin 4) ℝ) 
  (h : A * B = !![1, 0, -1, 0; 0, 1, 0, -1; -1, 0, 1, 0; 0, -1, 0, 1]) :
  B * A = !![2, 0; 0, 2] := by sorry

end NUMINAMATH_CALUDE_BA_is_2I_l3476_347660


namespace NUMINAMATH_CALUDE_max_inscribed_sphere_volume_l3476_347689

theorem max_inscribed_sphere_volume (cone_base_diameter : ℝ) (cone_volume : ℝ) 
  (h_diameter : cone_base_diameter = 12)
  (h_volume : cone_volume = 96 * Real.pi) : 
  let cone_radius : ℝ := cone_base_diameter / 2
  let cone_height : ℝ := 3 * cone_volume / (Real.pi * cone_radius^2)
  let cone_slant_height : ℝ := Real.sqrt (cone_radius^2 + cone_height^2)
  let sphere_radius : ℝ := cone_radius * cone_height / (cone_radius + cone_height + cone_slant_height)
  let sphere_volume : ℝ := 4 / 3 * Real.pi * sphere_radius^3
  sphere_volume = 36 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_max_inscribed_sphere_volume_l3476_347689


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3476_347676

theorem sqrt_simplification : Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3476_347676


namespace NUMINAMATH_CALUDE_tan_20_plus_4sin_20_equals_sqrt_3_l3476_347630

theorem tan_20_plus_4sin_20_equals_sqrt_3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_20_plus_4sin_20_equals_sqrt_3_l3476_347630


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3476_347613

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- subset relation for a line in a plane
variable (perpendicular : Line → Line → Prop)  -- perpendicular relation between lines
variable (perpendicularToPlane : Line → Plane → Prop)  -- perpendicular relation between a line and a plane
variable (parallel : Plane → Plane → Prop)  -- parallel relation between planes

-- State the theorem
theorem sufficient_but_not_necessary
  (a b : Line) (α β : Plane)
  (h1 : subset a α)
  (h2 : perpendicularToPlane b β)
  (h3 : parallel α β) :
  perpendicular a b ∧
  ¬(∀ (a b : Line) (α β : Plane),
    perpendicular a b →
    subset a α ∧ perpendicularToPlane b β ∧ parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3476_347613


namespace NUMINAMATH_CALUDE_percentage_difference_l3476_347643

theorem percentage_difference : (56 * 0.50) - (50 * 0.30) = 13 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l3476_347643


namespace NUMINAMATH_CALUDE_triangle_theorem_l3476_347663

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * (Real.sqrt 3 * Real.tan t.B - 1) = 
        (t.b * Real.cos t.A / Real.cos t.B) + (t.c * Real.cos t.A / Real.cos t.C))
  (h2 : t.a + t.b + t.c = 20)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3)
  (h4 : t.a > t.b) :
  t.C = Real.pi / 3 ∧ t.a = 8 ∧ t.b = 5 ∧ t.c = 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3476_347663


namespace NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l3476_347644

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point is the incenter of a triangle -/
def is_incenter (I : Point) (t : Triangle) : Prop := sorry

/-- Checks if a point is the foot of the altitude from C to AB -/
def is_altitude_foot (H : Point) (t : Triangle) : Prop := sorry

/-- Checks if a point is the excenter opposite to C -/
def is_excenter_C (I_C : Point) (t : Triangle) : Prop := sorry

/-- Checks if the excenter touches side AB and extensions of AC and BC -/
def excenter_touches_sides (I_C : Point) (t : Triangle) : Prop := sorry

theorem triangle_reconstruction_uniqueness 
  (I H I_C : Point) : 
  ∃! t : Triangle, 
    is_incenter I t ∧ 
    is_altitude_foot H t ∧ 
    is_excenter_C I_C t ∧ 
    excenter_touches_sides I_C t :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l3476_347644


namespace NUMINAMATH_CALUDE_abs_d_equals_three_l3476_347635

/-- A polynomial with integer coefficients that has 3+i as a root -/
def f (a b c d : ℤ) : ℂ → ℂ := λ x => a*x^5 + b*x^4 + c*x^3 + d*x^2 + b*x + a

/-- The theorem stating that under given conditions, |d| = 3 -/
theorem abs_d_equals_three (a b c d : ℤ) : 
  f a b c d (3 + I) = 0 → 
  Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs = 1 → 
  d.natAbs = 3 := by sorry

end NUMINAMATH_CALUDE_abs_d_equals_three_l3476_347635


namespace NUMINAMATH_CALUDE_sum_of_acute_angles_not_always_obtuse_l3476_347687

-- Define what an acute angle is
def is_acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define what an obtuse angle is
def is_obtuse_angle (α : Real) : Prop := Real.pi / 2 < α ∧ α < Real.pi

-- Theorem stating that the sum of two acute angles is not always obtuse
theorem sum_of_acute_angles_not_always_obtuse :
  ∃ (α β : Real), is_acute_angle α ∧ is_acute_angle β ∧ ¬is_obtuse_angle (α + β) :=
sorry

end NUMINAMATH_CALUDE_sum_of_acute_angles_not_always_obtuse_l3476_347687


namespace NUMINAMATH_CALUDE_smallest_M_inequality_l3476_347684

theorem smallest_M_inequality (a b c : ℝ) :
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ N : ℝ, (∀ x y z : ℝ, |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) →
    M ≤ N ∧ |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_M_inequality_l3476_347684


namespace NUMINAMATH_CALUDE_problem_solution_l3476_347666

theorem problem_solution : 
  3.2 * 2.25 - (5 * 0.85) / 2.5 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3476_347666


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3476_347695

theorem point_in_second_quadrant (a : ℤ) : 
  (2*a + 1 < 0) ∧ (2 + a > 0) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3476_347695


namespace NUMINAMATH_CALUDE_min_value_expression_l3476_347652

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 3) :
  a^2 + 8*a*b + 32*b^2 + 24*b*c + 8*c^2 ≥ 72 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 3 ∧
    a₀^2 + 8*a₀*b₀ + 32*b₀^2 + 24*b₀*c₀ + 8*c₀^2 = 72 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3476_347652


namespace NUMINAMATH_CALUDE_tims_number_l3476_347650

theorem tims_number (n : ℕ) : 
  (∃ k l : ℕ, n = 9 * k - 2 ∧ n = 8 * l - 4) ∧ 
  n < 150 ∧ 
  (∀ m : ℕ, (∃ p q : ℕ, m = 9 * p - 2 ∧ m = 8 * q - 4) ∧ m < 150 → m ≤ n) →
  n = 124 := by
sorry

end NUMINAMATH_CALUDE_tims_number_l3476_347650


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_equals_twice_sum_iff_zero_l3476_347678

theorem sqrt_sum_squares_equals_twice_sum_iff_zero (a b : ℝ) : 
  a ≥ 0 → b ≥ 0 → (Real.sqrt (a^2 + b^2) = 2 * (a + b) ↔ a = 0 ∧ b = 0) := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_equals_twice_sum_iff_zero_l3476_347678


namespace NUMINAMATH_CALUDE_inequality_solution_l3476_347621

theorem inequality_solution (x : ℝ) : (x + 10) / (x^2 + 2*x + 5) ≥ 0 ↔ x ≥ -10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3476_347621


namespace NUMINAMATH_CALUDE_smallest_value_complex_expression_l3476_347685

theorem smallest_value_complex_expression (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_fourth : ω^4 = 1)
  (h_omega_not_one : ω ≠ 1) :
  ∃ (min_val : ℝ), 
    (∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z), 
      Complex.abs (x + y * ω + z * ω^3) ≥ min_val) ∧
    (∃ (p q r : ℤ) (h_pqr_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r), 
      Complex.abs (p + q * ω + r * ω^3) = min_val) ∧
    min_val = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_complex_expression_l3476_347685


namespace NUMINAMATH_CALUDE_angle_457_properties_l3476_347647

-- Define the set of angles with the same terminal side as -457°
def same_terminal_side (β : ℝ) : Prop :=
  ∃ k : ℤ, β = k * 360 - 457

-- Define the third quadrant
def third_quadrant (θ : ℝ) : Prop :=
  180 < θ % 360 ∧ θ % 360 < 270

-- Theorem statement
theorem angle_457_properties :
  (∀ β, same_terminal_side β ↔ ∃ k : ℤ, β = k * 360 - 457) ∧
  third_quadrant (-457) := by
  sorry

end NUMINAMATH_CALUDE_angle_457_properties_l3476_347647


namespace NUMINAMATH_CALUDE_three_planes_theorem_l3476_347658

-- Define the two equations
def equation_cubic (x y z : ℝ) : Prop :=
  x^3 + y^3 + z^3 = (x + y + z)^3

def equation_quintic (x y z : ℝ) : Prop :=
  x^5 + y^5 + z^5 = (x + y + z)^5

-- State the theorem
theorem three_planes_theorem :
  ∀ (x y z : ℝ),
    (equation_cubic x y z → (x + y) * (y + z) * (z + x) = 0) ∧
    (equation_quintic x y z → (x + y) * (y + z) * (z + x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_three_planes_theorem_l3476_347658


namespace NUMINAMATH_CALUDE_evaluate_expression_l3476_347654

theorem evaluate_expression : (3200 - 3131)^2 / 121 = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3476_347654


namespace NUMINAMATH_CALUDE_series_value_l3476_347631

def series_term (n : ℕ) : ℤ := n * (n + 1) - (n + 1) * (n + 2)

def series_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => series_sum n + series_term (n + 1)

theorem series_value : series_sum 2000 = 2004002 := by
  sorry

end NUMINAMATH_CALUDE_series_value_l3476_347631


namespace NUMINAMATH_CALUDE_circle_center_transformation_l3476_347651

/-- Reflects a point about the line y=x --/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Translates a point by a given vector --/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

/-- The main theorem --/
theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (8, -3)
  let reflected_center := reflect_about_y_eq_x initial_center
  let translation_vector : ℝ × ℝ := (4, 2)
  let final_center := translate reflected_center translation_vector
  final_center = (1, 10) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l3476_347651


namespace NUMINAMATH_CALUDE_exists_m_composite_l3476_347698

theorem exists_m_composite (n : ℕ) : ∃ m : ℕ, ∃ k : ℕ, k > 1 ∧ k < n * m + 1 ∧ (n * m + 1) % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_composite_l3476_347698


namespace NUMINAMATH_CALUDE_total_tape_is_870_l3476_347641

/-- Calculates the tape length for a side, including overlap -/
def tape_length (side : ℕ) : ℕ := side + 2

/-- Calculates the tape needed for a single box -/
def box_tape (length width : ℕ) : ℕ :=
  tape_length length + 2 * tape_length width

/-- The total tape needed for all boxes -/
def total_tape : ℕ :=
  5 * box_tape 30 15 +
  2 * box_tape 40 40 +
  3 * box_tape 50 20

theorem total_tape_is_870 : total_tape = 870 := by
  sorry

end NUMINAMATH_CALUDE_total_tape_is_870_l3476_347641


namespace NUMINAMATH_CALUDE_eraser_price_l3476_347617

/-- Proves that the price of an eraser is $1 given the problem conditions --/
theorem eraser_price (pencils_sold : ℕ) (total_earnings : ℝ) 
  (h1 : pencils_sold = 20)
  (h2 : total_earnings = 80)
  (h3 : ∀ p : ℝ, p > 0 → 
    pencils_sold * p + 2 * pencils_sold * (p / 2) = total_earnings) :
  ∃ (pencil_price : ℝ), 
    pencil_price > 0 ∧ 
    pencil_price / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eraser_price_l3476_347617


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l3476_347699

theorem consecutive_numbers_sum (n : ℕ) :
  (n + 1) + (n + 2) + (n + 3) = 2 * (n + (n - 1) + (n - 2)) →
  n + 3 = 7 ∧ (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) = 27 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l3476_347699


namespace NUMINAMATH_CALUDE_seventh_observation_value_l3476_347683

theorem seventh_observation_value (initial_count : Nat) (initial_average : ℝ) (new_average : ℝ) :
  initial_count = 6 →
  initial_average = 14 →
  new_average = 13 →
  (initial_count * initial_average + 7) / (initial_count + 1) = new_average →
  7 = (initial_count + 1) * new_average - initial_count * initial_average :=
by sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l3476_347683


namespace NUMINAMATH_CALUDE_units_digit_of_composite_product_l3476_347606

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_composite_product :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_composite_product_l3476_347606


namespace NUMINAMATH_CALUDE_quadratic_sqrt2_closure_l3476_347690

-- Define a structure for numbers of the form a + b√2
structure QuadraticSqrt2 where
  a : ℚ
  b : ℚ

-- Define addition for QuadraticSqrt2
def add (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a + y.a, x.b + y.b⟩

-- Define subtraction for QuadraticSqrt2
def sub (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a - y.a, x.b - y.b⟩

-- Define multiplication for QuadraticSqrt2
def mul (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a * y.a + 2 * x.b * y.b, x.a * y.b + x.b * y.a⟩

-- Define division for QuadraticSqrt2
def div (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  let denom := y.a * y.a - 2 * y.b * y.b
  ⟨(x.a * y.a - 2 * x.b * y.b) / denom, (x.b * y.a - x.a * y.b) / denom⟩

theorem quadratic_sqrt2_closure (x y : QuadraticSqrt2) (h : y.a * y.a ≠ 2 * y.b * y.b) :
  (∃ (z : QuadraticSqrt2), add x y = z) ∧
  (∃ (z : QuadraticSqrt2), sub x y = z) ∧
  (∃ (z : QuadraticSqrt2), mul x y = z) ∧
  (∃ (z : QuadraticSqrt2), div x y = z) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sqrt2_closure_l3476_347690


namespace NUMINAMATH_CALUDE_susan_remaining_money_l3476_347655

def susan_spending (initial_amount games_multiplier snacks_cost souvenir_cost : ℕ) : ℕ :=
  initial_amount - (snacks_cost + games_multiplier * snacks_cost + souvenir_cost)

theorem susan_remaining_money :
  susan_spending 80 3 15 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_susan_remaining_money_l3476_347655


namespace NUMINAMATH_CALUDE_birds_in_tree_l3476_347657

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3476_347657


namespace NUMINAMATH_CALUDE_syrup_problem_l3476_347611

/-- Represents a container with a certain volume of liquid --/
structure Container where
  syrup : ℝ
  water : ℝ

/-- The state of the three containers --/
structure ContainerState where
  a : Container
  b : Container
  c : Container

/-- Represents a pouring action --/
inductive PourAction
  | PourAll : Fin 3 → Fin 3 → PourAction
  | Equalize : Fin 3 → Fin 3 → PourAction
  | PourToSink : Fin 3 → PourAction

/-- Defines if a given sequence of actions is valid --/
def isValidActionSequence (initialState : ContainerState) (actions : List PourAction) : Prop :=
  sorry

/-- Defines if a final state has 10L of 30% syrup in one container --/
def hasTenLitersThirtyPercentSyrup (state : ContainerState) : Prop :=
  sorry

/-- The main theorem to prove --/
theorem syrup_problem (n : ℕ) :
  (∃ (actions : List PourAction),
    isValidActionSequence
      ⟨⟨3, 0⟩, ⟨0, n⟩, ⟨0, 0⟩⟩
      actions ∧
    hasTenLitersThirtyPercentSyrup
      (actions.foldl (λ state action => sorry) ⟨⟨3, 0⟩, ⟨0, n⟩, ⟨0, 0⟩⟩)) ↔
  ∃ (k : ℕ), n = 3 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_syrup_problem_l3476_347611


namespace NUMINAMATH_CALUDE_oranges_picked_l3476_347672

theorem oranges_picked (michaela_full : ℕ) (cassandra_full : ℕ) (remaining : ℕ) : 
  michaela_full = 20 → 
  cassandra_full = 2 * michaela_full → 
  remaining = 30 → 
  michaela_full + cassandra_full + remaining = 90 := by
  sorry

end NUMINAMATH_CALUDE_oranges_picked_l3476_347672


namespace NUMINAMATH_CALUDE_four_thirds_of_twelve_fifths_l3476_347625

theorem four_thirds_of_twelve_fifths :
  (4 : ℚ) / 3 * (12 : ℚ) / 5 = (16 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_twelve_fifths_l3476_347625


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l3476_347667

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l3476_347667


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3476_347659

theorem polynomial_value_theorem (a : ℝ) : 
  2 * a^2 + 3 * a + 1 = 6 → -6 * a^2 - 9 * a + 8 = -7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3476_347659


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3476_347671

theorem two_numbers_problem (x y : ℚ) : 
  (4 * y = 9 * x) → 
  (y - x = 12) → 
  y = 108 / 5 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3476_347671


namespace NUMINAMATH_CALUDE_sasha_max_quarters_l3476_347697

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 32 / 10

/-- The maximum number of quarters Sasha can have -/
def max_quarters : ℕ := 10

theorem sasha_max_quarters :
  ∀ q : ℕ,
  (q : ℚ) * (quarter_value + nickel_value) ≤ total_amount →
  q ≤ max_quarters :=
by sorry

end NUMINAMATH_CALUDE_sasha_max_quarters_l3476_347697


namespace NUMINAMATH_CALUDE_twenty_multi_painted_cubes_l3476_347696

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  top_painted : Bool
  sides_painted : Bool
  bottom_painted : Bool

/-- Counts the number of unit cubes with at least two painted faces -/
def count_multi_painted_cubes (cube : PaintedCube) : ℕ :=
  sorry

/-- The main theorem -/
theorem twenty_multi_painted_cubes :
  let cube : PaintedCube := {
    size := 4,
    top_painted := true,
    sides_painted := true,
    bottom_painted := false
  }
  count_multi_painted_cubes cube = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_multi_painted_cubes_l3476_347696


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3476_347649

def number_of_knights : ℕ := 30
def chosen_knights : ℕ := 4

def probability_adjacent_knights : ℚ :=
  1 - (number_of_knights - chosen_knights + 1) * (number_of_knights - chosen_knights - 1) * (number_of_knights - chosen_knights - 3) * (number_of_knights - chosen_knights - 5) / (number_of_knights.choose chosen_knights)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 53 / 85 := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l3476_347649


namespace NUMINAMATH_CALUDE_equation_holds_iff_conditions_l3476_347680

theorem equation_holds_iff_conditions (a b c : ℤ) :
  a * (a - b) + b * (b - c) + c * (c - a) = 2 ↔ 
  ((a = b ∧ b = c + 1) ∨ (a = c ∧ b - 1 = c)) := by
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_conditions_l3476_347680


namespace NUMINAMATH_CALUDE_largest_angle_of_obtuse_isosceles_triangle_l3476_347604

-- Define the triangle PQR
structure Triangle (P Q R : Point) where
  -- Add any necessary fields

-- Define the properties of the triangle
def isObtuse (t : Triangle P Q R) : Prop := sorry
def isIsosceles (t : Triangle P Q R) : Prop := sorry
def angleMeasure (p : Point) (t : Triangle P Q R) : ℝ := sorry
def largestAngle (t : Triangle P Q R) : ℝ := sorry

-- Theorem statement
theorem largest_angle_of_obtuse_isosceles_triangle 
  (P Q R : Point) (t : Triangle P Q R)
  (h_obtuse : isObtuse t)
  (h_isosceles : isIsosceles t)
  (h_angle_P : angleMeasure P t = 30) :
  largestAngle t = 120 := by sorry

end NUMINAMATH_CALUDE_largest_angle_of_obtuse_isosceles_triangle_l3476_347604


namespace NUMINAMATH_CALUDE_complex_modulus_l3476_347642

theorem complex_modulus (z : ℂ) (h : z * (2 + Complex.I) = 1 + 7 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3476_347642


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l3476_347616

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 : ℝ) / 4 * (p + q + r + s) = 15) : 
  (p + q + r + s) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l3476_347616


namespace NUMINAMATH_CALUDE_runner_lap_time_l3476_347633

/-- Proves that given a 400-meter track, a runner completing 3 laps with the first lap in 70 seconds
    and an average speed of 5 m/s for the entire run, the time for each of the second and third laps
    is 85 seconds. -/
theorem runner_lap_time (track_length : ℝ) (num_laps : ℕ) (first_lap_time : ℝ) (avg_speed : ℝ) :
  track_length = 400 →
  num_laps = 3 →
  first_lap_time = 70 →
  avg_speed = 5 →
  ∃ (second_third_lap_time : ℝ),
    second_third_lap_time = 85 ∧
    (track_length * num_laps) / avg_speed = first_lap_time + 2 * second_third_lap_time :=
by sorry

end NUMINAMATH_CALUDE_runner_lap_time_l3476_347633


namespace NUMINAMATH_CALUDE_correct_calculation_result_l3476_347665

theorem correct_calculation_result (x : ℤ) (h : x - 63 = 8) : x * 8 = 568 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l3476_347665


namespace NUMINAMATH_CALUDE_trigonometric_sum_equality_l3476_347603

theorem trigonometric_sum_equality (θ φ : Real) 
  (h : (Real.cos θ)^6 / (Real.cos φ)^2 + (Real.sin θ)^6 / (Real.sin φ)^2 = 1) :
  ∃ (x : Real), x = (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 ∧ 
  (∀ (y : Real), y = (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 → y ≤ x) ∧
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equality_l3476_347603


namespace NUMINAMATH_CALUDE_a_share_of_profit_l3476_347627

/-- Calculates the share of profit for a partner in a business partnership --/
def calculateShareOfProfit (investmentA investmentB investmentC totalProfit : ℕ) : ℕ :=
  let totalInvestment := investmentA + investmentB + investmentC
  (investmentA * totalProfit) / totalInvestment

/-- Theorem stating that A's share of the profit is 4260 --/
theorem a_share_of_profit :
  calculateShareOfProfit 6300 4200 10500 14200 = 4260 := by
  sorry

#eval calculateShareOfProfit 6300 4200 10500 14200

end NUMINAMATH_CALUDE_a_share_of_profit_l3476_347627


namespace NUMINAMATH_CALUDE_total_seashells_l3476_347623

theorem total_seashells (joan_shells jessica_shells : ℕ) : 
  joan_shells = 6 → jessica_shells = 8 → joan_shells + jessica_shells = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l3476_347623
