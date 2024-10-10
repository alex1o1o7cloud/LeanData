import Mathlib

namespace nancy_added_pencils_l2665_266572

/-- The number of pencils Nancy placed in the drawer -/
def pencils_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Nancy added 45 pencils to the drawer -/
theorem nancy_added_pencils : pencils_added 27 72 = 45 := by
  sorry

end nancy_added_pencils_l2665_266572


namespace triangle_abc_right_angle_l2665_266597

theorem triangle_abc_right_angle (A B C : ℝ) (h1 : A = 30) (h2 : B = 60) : C = 90 := by
  sorry

end triangle_abc_right_angle_l2665_266597


namespace max_blocks_fit_l2665_266516

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the box dimensions -/
def box : Dimensions := ⟨5, 4, 6⟩

/-- Represents the block dimensions -/
def block : Dimensions := ⟨3, 3, 2⟩

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := 6

theorem max_blocks_fit :
  (volume box) / (volume block) ≥ max_blocks ∧
  max_blocks * (volume block) ≤ volume box ∧
  max_blocks * block.length ≤ box.length + block.length - 1 ∧
  max_blocks * block.width ≤ box.width + block.width - 1 ∧
  max_blocks * block.height ≤ box.height + block.height - 1 :=
by sorry

end max_blocks_fit_l2665_266516


namespace orange_cost_l2665_266514

theorem orange_cost (cost_three_dozen : ℝ) (h : cost_three_dozen = 28.20) :
  let cost_per_dozen : ℝ := cost_three_dozen / 3
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 47.00 := by
sorry

end orange_cost_l2665_266514


namespace inequality_solution_implies_k_value_l2665_266586

theorem inequality_solution_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end inequality_solution_implies_k_value_l2665_266586


namespace original_salary_l2665_266568

def salary_change (x : ℝ) : ℝ := (1 + 0.1) * (1 - 0.05) * x

theorem original_salary : 
  ∃ (x : ℝ), salary_change x = 2090 ∧ x = 2000 :=
by sorry

end original_salary_l2665_266568


namespace quadratic_inequality_solution_l2665_266569

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that a + b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a + b = -5 :=
by sorry

end quadratic_inequality_solution_l2665_266569


namespace quadratic_equation_magnitude_l2665_266502

theorem quadratic_equation_magnitude (z : ℂ) : 
  z^2 - 12*z + 157 = 0 → ∃! r : ℝ, (Complex.abs z = r ∧ r = Real.sqrt 157) :=
by sorry

end quadratic_equation_magnitude_l2665_266502


namespace pocket_probabilities_l2665_266535

/-- Represents the number of balls in the pocket -/
def total_balls : ℕ := 5

/-- Represents the number of white balls in the pocket -/
def white_balls : ℕ := 3

/-- Represents the number of black balls in the pocket -/
def black_balls : ℕ := 2

/-- Represents the number of balls drawn at once -/
def drawn_balls : ℕ := 2

/-- The total number of ways to draw 2 balls from 5 balls -/
def total_events : ℕ := Nat.choose total_balls drawn_balls

/-- The probability of drawing two white balls -/
def prob_two_white : ℚ := (Nat.choose white_balls drawn_balls : ℚ) / total_events

/-- The probability of drawing one black and one white ball -/
def prob_one_black_one_white : ℚ := (white_balls * black_balls : ℚ) / total_events

theorem pocket_probabilities :
  total_events = 10 ∧
  prob_two_white = 3 / 10 ∧
  prob_one_black_one_white = 3 / 5 := by
  sorry

end pocket_probabilities_l2665_266535


namespace sticker_pages_l2665_266506

theorem sticker_pages (stickers_per_page : ℕ) (total_stickers : ℕ) (h1 : stickers_per_page = 10) (h2 : total_stickers = 220) :
  total_stickers / stickers_per_page = 22 := by
  sorry

end sticker_pages_l2665_266506


namespace square_area_error_l2665_266512

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end square_area_error_l2665_266512


namespace power_of_negative_square_l2665_266571

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by sorry

end power_of_negative_square_l2665_266571


namespace expression_factorization_l2665_266591

theorem expression_factorization (x : ℝ) :
  (12 * x^4 + 34 * x^3 + 45 * x - 6) - (3 * x^4 - 7 * x^3 + 8 * x - 6) = x * (9 * x^3 + 41 * x^2 + 37) := by
  sorry

end expression_factorization_l2665_266591


namespace sqrt_1936_div_11_l2665_266534

theorem sqrt_1936_div_11 : Real.sqrt 1936 / 11 = 4 := by sorry

end sqrt_1936_div_11_l2665_266534


namespace approximation_accuracy_l2665_266526

/-- The actual number of students --/
def actual_number : ℕ := 76500

/-- The approximate number in scientific notation --/
def approximate_number : ℝ := 7.7 * 10^4

/-- Definition of accuracy to thousands place --/
def accurate_to_thousands (x y : ℝ) : Prop :=
  ∃ k : ℤ, x = k * 1000 ∧ |y - x| < 500

/-- Theorem stating the approximation is accurate to the thousands place --/
theorem approximation_accuracy :
  accurate_to_thousands (↑actual_number) approximate_number :=
sorry

end approximation_accuracy_l2665_266526


namespace arithmetic_sequence_sum_l2665_266562

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 29) →
  (a 3 + a 6 + a 9 = 13) :=
by
  sorry

end arithmetic_sequence_sum_l2665_266562


namespace a_gt_abs_b_sufficient_not_necessary_l2665_266576

theorem a_gt_abs_b_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) := by
sorry

end a_gt_abs_b_sufficient_not_necessary_l2665_266576


namespace opposite_of_two_thirds_l2665_266540

theorem opposite_of_two_thirds :
  (-(2 : ℚ) / 3) = (-1 : ℚ) * (2 : ℚ) / 3 := by sorry

end opposite_of_two_thirds_l2665_266540


namespace problem_1_l2665_266554

theorem problem_1 : |-3| - 2 - (-6) / (-2) = -2 := by
  sorry

end problem_1_l2665_266554


namespace sum_of_ages_l2665_266536

/-- Represents the ages of Xavier and Yasmin -/
structure Ages where
  xavier : ℕ
  yasmin : ℕ

/-- The current ages of Xavier and Yasmin satisfy the given conditions -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.xavier = 2 * ages.yasmin ∧ ages.xavier + 6 = 30

/-- Theorem: The sum of Xavier's and Yasmin's current ages is 36 -/
theorem sum_of_ages (ages : Ages) (h : satisfies_conditions ages) : 
  ages.xavier + ages.yasmin = 36 := by
  sorry


end sum_of_ages_l2665_266536


namespace food_bank_donation_ratio_l2665_266590

/-- Proves the ratio of food donations in the second week to the first week -/
theorem food_bank_donation_ratio :
  let first_week_donation : ℝ := 40
  let second_week_multiple : ℝ := x
  let total_donation : ℝ := first_week_donation + first_week_donation * second_week_multiple
  let remaining_percentage : ℝ := 0.3
  let remaining_food : ℝ := 36
  remaining_percentage * total_donation = remaining_food →
  second_week_multiple = 2 := by
  sorry

end food_bank_donation_ratio_l2665_266590


namespace greatest_divisor_with_remainders_l2665_266500

theorem greatest_divisor_with_remainders : Nat.gcd (1657 - 6) (2037 - 5) = 127 := by
  sorry

end greatest_divisor_with_remainders_l2665_266500


namespace floor_inequality_and_factorial_divisibility_l2665_266530

theorem floor_inequality_and_factorial_divisibility 
  (x y : ℝ) (m n : ℕ+) 
  (hx : x ≥ 0) (hy : y ≥ 0) : 
  (⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋) ∧ 
  (∃ k : ℕ, k * (m.val.factorial * n.val.factorial * (3 * m.val + n.val).factorial * (3 * n.val + m.val).factorial) = 
   (5 * m.val).factorial * (5 * n.val).factorial) :=
sorry

end floor_inequality_and_factorial_divisibility_l2665_266530


namespace modular_inverse_15_mod_16_l2665_266520

theorem modular_inverse_15_mod_16 : ∃ x : ℤ, (15 * x) % 16 = 1 :=
by
  use 15
  sorry

end modular_inverse_15_mod_16_l2665_266520


namespace exists_polygon_9_exists_polygon_8_l2665_266507

/-- A polygon is represented as a list of points in the plane -/
def Polygon := List (ℝ × ℝ)

/-- Check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

/-- Property: each side of the polygon lies on a line containing at least one additional vertex -/
def satisfiesProperty (poly : Polygon) : Prop :=
  ∀ i j : Fin poly.length,
    i ≠ j →
    ∃ k : Fin poly.length, k ≠ i ∧ k ≠ j ∧
      collinear (poly.get i) (poly.get j) (poly.get k)

/-- Theorem: There exists a polygon with at most 9 vertices satisfying the property -/
theorem exists_polygon_9 : ∃ poly : Polygon, poly.length ≤ 9 ∧ satisfiesProperty poly :=
  sorry

/-- Theorem: There exists a polygon with at most 8 vertices satisfying the property -/
theorem exists_polygon_8 : ∃ poly : Polygon, poly.length ≤ 8 ∧ satisfiesProperty poly :=
  sorry

end exists_polygon_9_exists_polygon_8_l2665_266507


namespace grape_to_fruit_ratio_l2665_266596

def red_apples : ℕ := 9
def green_apples : ℕ := 4
def grape_bunches : ℕ := 3
def grapes_per_bunch : ℕ := 15
def yellow_bananas : ℕ := 6
def orange_oranges : ℕ := 2
def kiwis : ℕ := 5
def blueberries : ℕ := 30

def total_grapes : ℕ := grape_bunches * grapes_per_bunch

def total_fruits : ℕ := red_apples + green_apples + total_grapes + yellow_bananas + orange_oranges + kiwis + blueberries

theorem grape_to_fruit_ratio :
  (total_grapes : ℚ) / (total_fruits : ℚ) = 45 / 101 := by
  sorry

end grape_to_fruit_ratio_l2665_266596


namespace ratio_calculation_l2665_266532

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 ∧ B / C = 1/5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
sorry

end ratio_calculation_l2665_266532


namespace range_of_m_l2665_266503

-- Define the inequality system
def inequality_system (x m : ℝ) : Prop :=
  x + 5 < 5*x + 1 ∧ x - m > 1

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x > 1

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∀ x, inequality_system x m ↔ solution_set x) →
  m ≤ 0 :=
by sorry

end range_of_m_l2665_266503


namespace broccoli_area_l2665_266537

/-- Represents the garden and broccoli production --/
structure BroccoliGarden where
  last_year_side : ℝ
  this_year_side : ℝ
  broccoli_increase : ℕ
  this_year_count : ℕ

/-- The conditions of the broccoli garden problem --/
def broccoli_conditions (g : BroccoliGarden) : Prop :=
  g.broccoli_increase = 79 ∧
  g.this_year_count = 1600 ∧
  g.this_year_side ^ 2 = g.this_year_count ∧
  g.this_year_side ^ 2 - g.last_year_side ^ 2 = g.broccoli_increase

/-- The theorem stating that each broccoli takes 1 square foot --/
theorem broccoli_area (g : BroccoliGarden) 
  (h : broccoli_conditions g) : 
  g.this_year_side ^ 2 / g.this_year_count = 1 := by
  sorry


end broccoli_area_l2665_266537


namespace norma_laundry_problem_l2665_266527

/-- The number of T-shirts Norma left in the washer -/
def t_shirts_left : ℕ := 9

/-- The number of sweaters Norma left in the washer -/
def sweaters_left : ℕ := 2 * t_shirts_left

/-- The total number of clothes Norma left in the washer -/
def total_left : ℕ := t_shirts_left + sweaters_left

/-- The number of sweaters Norma found when she returned -/
def sweaters_found : ℕ := 3

/-- The number of T-shirts Norma found when she returned -/
def t_shirts_found : ℕ := 3 * t_shirts_left

/-- The total number of clothes Norma found when she returned -/
def total_found : ℕ := sweaters_found + t_shirts_found

/-- The number of missing items -/
def missing_items : ℕ := total_left - total_found

theorem norma_laundry_problem : missing_items = 15 := by
  sorry

end norma_laundry_problem_l2665_266527


namespace container_emptying_l2665_266557

/-- Represents the state of the three containers -/
structure ContainerState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a valid transfer between containers -/
inductive Transfer : ContainerState → ContainerState → Prop where
  | ab {s t : ContainerState} : t.a = s.a + s.a ∧ t.b = s.b - s.a ∧ t.c = s.c → Transfer s t
  | ac {s t : ContainerState} : t.a = s.a + s.a ∧ t.b = s.b ∧ t.c = s.c - s.a → Transfer s t
  | ba {s t : ContainerState} : t.a = s.a - s.b ∧ t.b = s.b + s.b ∧ t.c = s.c → Transfer s t
  | bc {s t : ContainerState} : t.a = s.a ∧ t.b = s.b + s.b ∧ t.c = s.c - s.b → Transfer s t
  | ca {s t : ContainerState} : t.a = s.a - s.c ∧ t.b = s.b ∧ t.c = s.c + s.c → Transfer s t
  | cb {s t : ContainerState} : t.a = s.a ∧ t.b = s.b - s.c ∧ t.c = s.c + s.c → Transfer s t

/-- A sequence of transfers -/
def TransferSeq : ContainerState → ContainerState → Prop :=
  Relation.ReflTransGen Transfer

/-- The main theorem stating that it's always possible to empty a container -/
theorem container_emptying (initial : ContainerState) : 
  ∃ (final : ContainerState), TransferSeq initial final ∧ (final.a = 0 ∨ final.b = 0 ∨ final.c = 0) := by
  sorry

end container_emptying_l2665_266557


namespace horse_price_theorem_l2665_266579

/-- The sum of a geometric series with 32 terms, where the first term is 1
    and each subsequent term is twice the previous term, is 4294967295. -/
theorem horse_price_theorem :
  let n : ℕ := 32
  let a : ℕ := 1
  let r : ℕ := 2
  (a * (r^n - 1)) / (r - 1) = 4294967295 :=
by sorry

end horse_price_theorem_l2665_266579


namespace loom_weaving_rate_l2665_266509

/-- The rate at which an industrial loom weaves cloth, given the time and length of cloth woven. -/
theorem loom_weaving_rate (time : ℝ) (length : ℝ) (h : time = 195.3125 ∧ length = 25) :
  length / time = 0.128 := by
  sorry

end loom_weaving_rate_l2665_266509


namespace solution_system_equations_l2665_266505

theorem solution_system_equations (x y z : ℝ) : 
  x^2 + y^2 = 6*z ∧ 
  y^2 + z^2 = 6*x ∧ 
  z^2 + x^2 = 6*y → 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 3 ∧ y = 3 ∧ z = 3) := by
sorry

end solution_system_equations_l2665_266505


namespace point_movement_l2665_266566

/-- Given a point P in a 2D Cartesian coordinate system, moving it right and down
    results in a new point Q with the expected coordinates. -/
theorem point_movement (P : ℝ × ℝ) (right down : ℝ) (Q : ℝ × ℝ) :
  P = (-1, 2) →
  right = 2 →
  down = 3 →
  Q.1 = P.1 + right →
  Q.2 = P.2 - down →
  Q = (1, -1) := by
  sorry

end point_movement_l2665_266566


namespace trouser_original_price_l2665_266567

theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 50 ∧ discount_percentage = 0.5 ∧ sale_price = (1 - discount_percentage) * original_price →
  original_price = 100 := by
  sorry

end trouser_original_price_l2665_266567


namespace min_value_f_range_of_a_inequality_ln_exp_l2665_266501

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x * Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 3

-- Statement 1
theorem min_value_f (t : ℝ) (h : t > 0) :
  (∀ x ∈ Set.Icc t (t + 2), f x ≥ (-1 / Real.exp 1) ∧ (t < 1 / Real.exp 1 → f x > t * Real.log t)) ∧
  (∀ x ∈ Set.Icc t (t + 2), f x ≥ t * Real.log t ∧ (t ≥ 1 / Real.exp 1 → f x > -1 / Real.exp 1)) :=
sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∀ x > 0, 2 * f x ≥ g a x) ↔ a ≤ 4 :=
sorry

-- Statement 3
theorem inequality_ln_exp (x : ℝ) (h : x > 0) :
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end min_value_f_range_of_a_inequality_ln_exp_l2665_266501


namespace only_D_satisfies_all_preferences_l2665_266521

-- Define the set of movies
inductive Movie : Type
  | A | B | C | D | E

-- Define the preferences of each person
def xiao_zhao_preference (m : Movie) : Prop := m ≠ Movie.B
def xiao_zhang_preference (m : Movie) : Prop := m = Movie.B ∨ m = Movie.C ∨ m = Movie.D ∨ m = Movie.E
def xiao_li_preference (m : Movie) : Prop := m ≠ Movie.C
def xiao_liu_preference (m : Movie) : Prop := m ≠ Movie.E

-- Define a function that checks if a movie satisfies all preferences
def satisfies_all_preferences (m : Movie) : Prop :=
  xiao_zhao_preference m ∧
  xiao_zhang_preference m ∧
  xiao_li_preference m ∧
  xiao_liu_preference m

-- Theorem: D is the only movie that satisfies all preferences
theorem only_D_satisfies_all_preferences :
  ∀ m : Movie, satisfies_all_preferences m ↔ m = Movie.D :=
by sorry


end only_D_satisfies_all_preferences_l2665_266521


namespace marias_purse_value_l2665_266594

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of nickels in Maria's purse -/
def num_nickels : ℕ := 2

/-- The number of dimes in Maria's purse -/
def num_dimes : ℕ := 3

/-- The number of quarters in Maria's purse -/
def num_quarters : ℕ := 2

theorem marias_purse_value :
  (num_nickels * nickel_value + num_dimes * dime_value + num_quarters * quarter_value) * 100 / cents_per_dollar = 90 := by
  sorry

end marias_purse_value_l2665_266594


namespace z_value_proof_l2665_266561

theorem z_value_proof : 
  ∃ z : ℝ, ((2^5 : ℝ) * (9^2)) / (z * (3^5)) = 0.16666666666666666 → z = 64 := by
  sorry

end z_value_proof_l2665_266561


namespace age_difference_l2665_266528

/-- Given that B is currently 42 years old, and in 10 years A will be twice as old as B was 10 years ago, prove that A is 12 years older than B. -/
theorem age_difference (A B : ℕ) : B = 42 → A + 10 = 2 * (B - 10) → A - B = 12 := by
  sorry

end age_difference_l2665_266528


namespace nonagon_diagonals_l2665_266533

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex polygon with 9 sides -/
structure ConvexNonagon where
  sides : ℕ
  is_convex : Bool
  side_count_eq_9 : sides = 9

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals (n : ConvexNonagon) : num_diagonals_nonagon = 27 := by
  sorry

end nonagon_diagonals_l2665_266533


namespace tan_ratio_from_sin_sum_diff_l2665_266559

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8)
  (h2 : Real.sin (a - b) = 1/4) :
  Real.tan a / Real.tan b = 7/3 := by
sorry

end tan_ratio_from_sin_sum_diff_l2665_266559


namespace parabola_x_intercepts_l2665_266547

theorem parabola_x_intercepts :
  let f (x : ℝ) := -3 * x^2 + 4 * x - 1
  (∃ a b : ℝ, a ≠ b ∧ f a = 0 ∧ f b = 0) ∧
  (∀ x y z : ℝ, f x = 0 → f y = 0 → f z = 0 → x = y ∨ x = z ∨ y = z) := by
  sorry

end parabola_x_intercepts_l2665_266547


namespace inverse_matrices_product_l2665_266546

def inverse_matrices (x y z w : ℝ) : Prop :=
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![x, 3; 4, y]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, z; w, -5]
  A * B = 1

theorem inverse_matrices_product (x y z w : ℝ) 
  (h : inverse_matrices x y z w) : x * y * z * w = -5040/49 := by
  sorry

end inverse_matrices_product_l2665_266546


namespace g_difference_l2665_266511

/-- Given g(x) = 3x^2 + 4x + 5, prove that g(x + h) - g(x) = h(6x + 3h + 4) for all real x and h. -/
theorem g_difference (x h : ℝ) : 
  let g : ℝ → ℝ := λ t ↦ 3 * t^2 + 4 * t + 5
  g (x + h) - g x = h * (6 * x + 3 * h + 4) := by
  sorry

end g_difference_l2665_266511


namespace complex_magnitude_equality_l2665_266529

theorem complex_magnitude_equality (t : ℝ) (h : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 5 → t = 6 := by
  sorry

end complex_magnitude_equality_l2665_266529


namespace price_after_discounts_l2665_266515

/-- The original price of an article before discounts -/
def original_price : ℝ := 70.59

/-- The final price after discounts -/
def final_price : ℝ := 36

/-- The first discount rate -/
def discount1 : ℝ := 0.15

/-- The second discount rate -/
def discount2 : ℝ := 0.25

/-- The third discount rate -/
def discount3 : ℝ := 0.20

/-- Theorem stating that the original price results in the final price after applying the discounts -/
theorem price_after_discounts : 
  ∃ ε > 0, abs (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) - final_price) < ε :=
sorry

end price_after_discounts_l2665_266515


namespace friday_dressing_time_l2665_266525

/-- Represents the dressing times for each day of the week -/
structure DressingTimes where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average dressing time for the week -/
def weeklyAverage (times : DressingTimes) : ℚ :=
  (times.monday + times.tuesday + times.wednesday + times.thursday + times.friday) / 5

/-- The old average dressing time -/
def oldAverage : ℚ := 3

/-- The given dressing times for Monday through Thursday -/
def givenTimes : DressingTimes := {
  monday := 2
  tuesday := 4
  wednesday := 3
  thursday := 4
  friday := 0  -- We'll solve for this
}

theorem friday_dressing_time :
  ∃ (fridayTime : ℕ),
    let newTimes := { givenTimes with friday := fridayTime }
    weeklyAverage newTimes = oldAverage ∧ fridayTime = 2 := by
  sorry


end friday_dressing_time_l2665_266525


namespace circle_radius_relation_l2665_266592

theorem circle_radius_relation (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (π * x^2 = π * y^2) → (2 * π * x = 20 * π) → y / 2 = 5 := by
  sorry

end circle_radius_relation_l2665_266592


namespace geometric_sequence_product_l2665_266556

/-- Given a geometric sequence {a_n} where a₂ = 2 and a₆ = 8, 
    prove that a₃ * a₄ * a₅ = 64 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_a6 : a 6 = 8) : 
  a 3 * a 4 * a 5 = 64 := by
  sorry

end geometric_sequence_product_l2665_266556


namespace f_properties_l2665_266538

noncomputable section

def f (x : ℝ) := (Real.log x) / x

theorem f_properties :
  ∀ x > 0,
  (∃ y, f x = y ∧ x - y - 1 = 0) ∧ 
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, Real.exp 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (f (Real.exp 1) = (Real.exp 1)⁻¹) ∧
  (∀ x, x > 0 → f x ≤ (Real.exp 1)⁻¹) :=
by sorry

end

end f_properties_l2665_266538


namespace point_inside_circle_l2665_266563

/-- Given an ellipse and an equation, prove that the point formed by the equation's roots is inside a specific circle -/
theorem point_inside_circle (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → b > 0 → a > b → -- Ellipse conditions
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → x^2 ≤ a^2 ∧ y^2 ≤ b^2) → -- Ellipse equation
  c/a = 1/2 → -- Eccentricity
  x₁^2 + (a*x₁/b)^2 = a^2 → -- x₁ is on the ellipse
  x₂^2 + (a*x₂/b)^2 = a^2 → -- x₂ is on the ellipse
  a*x₁^2 + b*x₁ - c = 0 → -- x₁ is a root of the equation
  a*x₂^2 + b*x₂ - c = 0 → -- x₂ is a root of the equation
  x₁^2 + x₂^2 < 2 -- Point (x₁, x₂) is inside the circle x^2 + y^2 = 2
  := by sorry

end point_inside_circle_l2665_266563


namespace parallel_line_intersection_parallel_planes_intersection_l2665_266544

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line)

-- Theorem 1
theorem parallel_line_intersection 
  (l m : Line) (α β : Plane) :
  parallel_line_plane l α →
  subset l β →
  intersect α β = m →
  parallel l m := by sorry

-- Theorem 2
theorem parallel_planes_intersection 
  (l m : Line) (α β γ : Plane) :
  parallel_plane α β →
  intersect α γ = l →
  intersect β γ = m →
  parallel l m := by sorry

end parallel_line_intersection_parallel_planes_intersection_l2665_266544


namespace halfway_to_end_time_l2665_266523

/-- Two cars traveling in opposite directions with constant speeds --/
structure TwoCars where
  speed : ℝ
  distance : ℝ

/-- The state of the cars at a given time --/
def carState (cars : TwoCars) (t : ℝ) : ℝ × ℝ :=
  (cars.speed * t, cars.distance - cars.speed * t)

/-- The condition that after 1 hour, one car is halfway to the other --/
def halfwayAfterOneHour (cars : TwoCars) : Prop :=
  cars.speed = (cars.distance / 2)

/-- The time when one car is halfway between the other car and its starting point --/
def halfwayToEnd (cars : TwoCars) : ℝ :=
  2

/-- The main theorem --/
theorem halfway_to_end_time {cars : TwoCars} (h : halfwayAfterOneHour cars) :
  let (x, y) := carState cars (halfwayToEnd cars)
  x = (cars.distance - y) / 2 := by
  sorry


end halfway_to_end_time_l2665_266523


namespace car_trip_distance_theorem_l2665_266553

/-- Represents a segment of a car trip with speed and duration -/
structure TripSegment where
  speed : ℝ  -- Speed in miles per hour
  duration : ℝ  -- Duration in hours

/-- Calculates the distance traveled for a trip segment -/
def distance_traveled (segment : TripSegment) : ℝ :=
  segment.speed * segment.duration

/-- Represents a car trip with multiple segments -/
def CarTrip : Type := List TripSegment

/-- Calculates the total distance traveled for a car trip -/
def total_distance (trip : CarTrip) : ℝ :=
  trip.map distance_traveled |>.sum

theorem car_trip_distance_theorem (trip : CarTrip) : 
  trip = [
    { speed := 65, duration := 3 },
    { speed := 45, duration := 2 },
    { speed := 55, duration := 4 }
  ] → total_distance trip = 505 := by
  sorry

end car_trip_distance_theorem_l2665_266553


namespace quadratic_roots_l2665_266599

theorem quadratic_roots :
  ∃ (x₁ x₂ : ℝ), (x₁ = 2 ∧ x₂ = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end quadratic_roots_l2665_266599


namespace perimeter_pedal_ratio_l2665_266573

/-- A triangle in a 2D plane -/
structure Triangle where
  -- Define the triangle structure (you may need to adjust this based on your specific needs)
  -- For example, you could define it using three points or side lengths

/-- The pedal triangle of a given triangle -/
def pedal_triangle (t : Triangle) : Triangle :=
  sorry -- Definition of pedal triangle

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  sorry -- Definition of perimeter

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ :=
  sorry -- Definition of circumradius

/-- The inradius of a triangle -/
def inradius (t : Triangle) : ℝ :=
  sorry -- Definition of inradius

/-- Theorem: The ratio of a triangle's perimeter to its pedal triangle's perimeter
    is equal to the ratio of its circumradius to its inradius -/
theorem perimeter_pedal_ratio (t : Triangle) :
  (perimeter t) / (perimeter (pedal_triangle t)) = (circumradius t) / (inradius t) := by
  sorry

end perimeter_pedal_ratio_l2665_266573


namespace intersection_point_d_l2665_266560

theorem intersection_point_d (d : ℝ) : 
  (∀ x y : ℝ, (y = x + d ∧ x = -y + d) → (x = d - 1 ∧ y = d)) → d = 1 := by
  sorry

end intersection_point_d_l2665_266560


namespace abs_c_equals_181_l2665_266587

def f (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_equals_181 (a b c : ℤ) :
  (Nat.gcd (Nat.gcd (a.natAbs) (b.natAbs)) (c.natAbs) = 1) →
  (f a b c (3 + 2*Complex.I) = 0) →
  (c.natAbs = 181) :=
sorry

end abs_c_equals_181_l2665_266587


namespace arithmetic_equality_l2665_266517

theorem arithmetic_equality : 5 - 4 * 3 / 2 + 1 = 0 := by sorry

end arithmetic_equality_l2665_266517


namespace total_birds_l2665_266519

-- Define the number of geese
def geese : ℕ := 58

-- Define the number of ducks
def ducks : ℕ := 37

-- Theorem stating the total number of birds
theorem total_birds : geese + ducks = 95 := by
  sorry

end total_birds_l2665_266519


namespace fraction_subtraction_l2665_266548

theorem fraction_subtraction (m : ℝ) (h : m ≠ 1) : m / (1 - m) - 1 / (1 - m) = -1 := by
  sorry

end fraction_subtraction_l2665_266548


namespace percentage_problem_l2665_266564

theorem percentage_problem : ∃ P : ℚ, P * 30 = 0.25 * 16 + 2 := by
  sorry

end percentage_problem_l2665_266564


namespace base_four_of_85_l2665_266539

def base_four_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_four_of_85 :
  base_four_representation 85 = [1, 1, 1, 1] := by
  sorry

end base_four_of_85_l2665_266539


namespace original_price_calculation_l2665_266593

def selling_price : ℝ := 1220
def gain_percentage : ℝ := 45.23809523809524

theorem original_price_calculation :
  let original_price := selling_price / (1 + gain_percentage / 100)
  ∃ ε > 0, |original_price - 840| < ε :=
by sorry

end original_price_calculation_l2665_266593


namespace jill_walking_time_l2665_266552

/-- The time it takes Jill to walk to school given Dave's and Jill's walking parameters -/
theorem jill_walking_time (dave_steps_per_min : ℕ) (dave_step_length : ℕ) (dave_time : ℕ)
  (jill_steps_per_min : ℕ) (jill_step_length : ℕ) 
  (h1 : dave_steps_per_min = 80) (h2 : dave_step_length = 65) (h3 : dave_time = 20)
  (h4 : jill_steps_per_min = 120) (h5 : jill_step_length = 50) :
  (dave_steps_per_min * dave_step_length * dave_time : ℚ) / (jill_steps_per_min * jill_step_length) = 52/3 :=
by sorry

end jill_walking_time_l2665_266552


namespace sector_angle_l2665_266580

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, 
    prove that its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (θ : ℝ) 
  (h_area : (1/2) * θ * r^2 = 1)
  (h_perim : 2*r + θ*r = 4) : 
  θ = 2 := by
  sorry

end sector_angle_l2665_266580


namespace pencil_cost_2500_l2665_266551

/-- The cost of buying a certain number of pencils with a discount applied after a threshold -/
def pencil_cost (box_size : ℕ) (box_cost : ℚ) (total_pencils : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let unit_cost := box_cost / box_size
  let regular_cost := min total_pencils discount_threshold * unit_cost
  let discounted_pencils := max (total_pencils - discount_threshold) 0
  let discounted_cost := discounted_pencils * (unit_cost * (1 - discount_rate))
  regular_cost + discounted_cost

theorem pencil_cost_2500 :
  pencil_cost 200 50 2500 1000 (1/10) = 587.5 := by
  sorry

end pencil_cost_2500_l2665_266551


namespace min_value_of_expression_l2665_266524

theorem min_value_of_expression (a : ℝ) (b : ℝ) (h : b > 0) :
  ∃ (min : ℝ), min = 2 * (1 - Real.log 2)^2 ∧
  ∀ (x y : ℝ) (hy : y > 0), ((1/2 * Real.exp x - Real.log (2*y))^2 + (x - y)^2) ≥ min :=
by sorry

end min_value_of_expression_l2665_266524


namespace chess_tournament_games_l2665_266513

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 6 → total_games = 12 → (n * (n - 1)) / 2 = total_games → n - 1 = 5 :=
by
  sorry

#check chess_tournament_games

end chess_tournament_games_l2665_266513


namespace retail_price_is_1_04a_l2665_266550

/-- The retail price of a washing machine after markup and discount -/
def retail_price (a : ℝ) : ℝ :=
  a * (1 + 0.3) * (1 - 0.2)

/-- Theorem stating that the retail price is 1.04 times the initial cost -/
theorem retail_price_is_1_04a (a : ℝ) : retail_price a = 1.04 * a := by
  sorry

end retail_price_is_1_04a_l2665_266550


namespace arithmetic_progression_term_position_l2665_266588

theorem arithmetic_progression_term_position
  (a d : ℚ)  -- first term and common difference
  (sum_two_terms : a + 11 * d + a + (x - 1) * d = 20)  -- sum of 12th and x-th term is 20
  (sum_ten_terms : 10 * a + 45 * d = 100)  -- sum of first 10 terms is 100
  (x : ℕ)  -- position of the other term
  : x = 8 := by
  sorry

end arithmetic_progression_term_position_l2665_266588


namespace gcd_eight_factorial_six_factorial_l2665_266542

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem gcd_eight_factorial_six_factorial :
  Nat.gcd (factorial 8) (factorial 6) = 720 := by
  sorry

end gcd_eight_factorial_six_factorial_l2665_266542


namespace expand_expression_l2665_266582

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l2665_266582


namespace term_2500_mod_7_l2665_266541

/-- Defines the sequence where the (2n)th positive integer appears n times
    and the (2n-1)th positive integer appears n+1 times -/
def sequence_term (k : ℕ) : ℕ := sorry

/-- The 2500th term of the sequence -/
def term_2500 : ℕ := sequence_term 2500

theorem term_2500_mod_7 : term_2500 % 7 = 1 := by sorry

end term_2500_mod_7_l2665_266541


namespace total_balloons_is_eighteen_l2665_266574

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := fred_balloons + sam_balloons + mary_balloons

theorem total_balloons_is_eighteen : total_balloons = 18 := by
  sorry

end total_balloons_is_eighteen_l2665_266574


namespace line_mb_product_l2665_266508

theorem line_mb_product (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + b) →  -- Line equation
  b = -3 →                      -- y-intercept
  5 = m * 3 + b →               -- Line passes through (3, 5)
  m * b = -8 := by sorry

end line_mb_product_l2665_266508


namespace fantasia_license_plates_l2665_266518

/-- The number of letters in the alphabet used for license plates. -/
def alphabet_size : ℕ := 26

/-- The number of digits used for license plates. -/
def digit_size : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Fantasia. -/
def total_license_plates : ℕ := alphabet_size ^ letter_positions * digit_size ^ digit_positions

theorem fantasia_license_plates :
  total_license_plates = 175760000 := by
  sorry

end fantasia_license_plates_l2665_266518


namespace x_twelfth_power_l2665_266589

theorem x_twelfth_power (x : ℂ) (h : x + 1/x = -Real.sqrt 3) : x^12 = 1 := by
  sorry

end x_twelfth_power_l2665_266589


namespace pentagon_fifth_angle_l2665_266543

/-- The sum of angles in a pentagon is 540 degrees -/
def pentagon_angle_sum : ℝ := 540

/-- The known angles of the pentagon -/
def known_angles : List ℝ := [130, 80, 105, 110]

/-- The measure of the unknown angle Q -/
def angle_q : ℝ := 115

/-- Theorem: In a pentagon with four known angles measuring 130°, 80°, 105°, and 110°, 
    the measure of the fifth angle is 115°. -/
theorem pentagon_fifth_angle :
  pentagon_angle_sum = (known_angles.sum + angle_q) :=
by sorry

end pentagon_fifth_angle_l2665_266543


namespace power_function_through_point_l2665_266570

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 3 = 27 → ∀ x : ℝ, f x = x^3 := by
  sorry

end power_function_through_point_l2665_266570


namespace tangent_angle_range_l2665_266545

theorem tangent_angle_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  let f : ℝ → ℝ := λ x => Real.log x + x / b
  let θ := Real.arctan (((1 / a) + (1 / b)) : ℝ)
  π / 4 ≤ θ ∧ θ < π / 2 := by
  sorry

end tangent_angle_range_l2665_266545


namespace min_even_integers_l2665_266555

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 28 →
  a + b + c + d = 45 →
  a + b + c + d + e + f = 63 →
  ∃ (n : ℕ), n ≥ 1 ∧ 
    ∀ (m : ℕ), (∃ (evens : Finset ℤ), evens.card = m ∧ 
      (∀ x ∈ evens, x % 2 = 0) ∧ 
      evens ⊆ {a, b, c, d, e, f}) → 
    n ≤ m :=
by sorry

end min_even_integers_l2665_266555


namespace sqrt_mixed_number_simplification_l2665_266504

theorem sqrt_mixed_number_simplification :
  Real.sqrt (12 + 1/9) = Real.sqrt 109 / 3 := by
  sorry

end sqrt_mixed_number_simplification_l2665_266504


namespace min_angle_in_special_right_triangle_l2665_266585

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Consecutive Fibonacci numbers -/
def consecutive_fib (a b : ℕ) : Prop :=
  ∃ n : ℕ, fib n = b ∧ fib (n + 1) = a

theorem min_angle_in_special_right_triangle :
  ∀ a b : ℕ,
    a > b →
    consecutive_fib a b →
    a + b = 100 →
    b ≥ 21 :=
sorry

end min_angle_in_special_right_triangle_l2665_266585


namespace corn_ratio_proof_l2665_266581

theorem corn_ratio_proof (marcel_corn : ℕ) (marcel_potatoes : ℕ) (dale_potatoes : ℕ) (total_vegetables : ℕ) :
  marcel_corn = 10 →
  marcel_potatoes = 4 →
  dale_potatoes = 8 →
  total_vegetables = 27 →
  ∃ (dale_corn : ℕ), 
    marcel_corn + marcel_potatoes + dale_corn + dale_potatoes = total_vegetables ∧
    dale_corn * 2 = marcel_corn :=
by sorry

end corn_ratio_proof_l2665_266581


namespace factorization_a_squared_minus_six_l2665_266549

theorem factorization_a_squared_minus_six (a : ℝ) :
  a^2 - 6 = (a + Real.sqrt 6) * (a - Real.sqrt 6) := by
  sorry

end factorization_a_squared_minus_six_l2665_266549


namespace diophantine_equation_solutions_l2665_266565

theorem diophantine_equation_solutions : 
  {(x, y) : ℕ × ℕ | 2 * x^2 + 2 * x * y - x + y = 2020} = {(0, 2020), (1, 673)} := by
  sorry

end diophantine_equation_solutions_l2665_266565


namespace one_white_ball_probability_l2665_266583

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (event total : ℕ) : ℚ := sorry

theorem one_white_ball_probability (bagA_white bagA_red bagB_white bagB_red : ℕ) 
  (h1 : bagA_white = 8)
  (h2 : bagA_red = 4)
  (h3 : bagB_white = 6)
  (h4 : bagB_red = 6) :
  probability 
    (choose bagA_white 1 * choose bagB_red 1 + choose bagA_red 1 * choose bagB_white 1)
    (choose (bagA_white + bagA_red) 1 * choose (bagB_white + bagB_red) 1) =
  probability 
    ((choose 8 1) * (choose 6 1) + (choose 4 1) * (choose 6 1))
    ((choose 12 1) * (choose 12 1)) :=
sorry

end one_white_ball_probability_l2665_266583


namespace binge_watching_duration_l2665_266595

/-- Proves that given a TV show with 90 episodes of 20 minutes each, and a viewing time of 2 hours per day, it will take 15 days to finish watching the entire show. -/
theorem binge_watching_duration (num_episodes : ℕ) (episode_length : ℕ) (daily_viewing_time : ℕ) : 
  num_episodes = 90 → 
  episode_length = 20 → 
  daily_viewing_time = 120 → 
  (num_episodes * episode_length) / daily_viewing_time = 15 := by
  sorry

#check binge_watching_duration

end binge_watching_duration_l2665_266595


namespace y_value_proof_l2665_266598

theorem y_value_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 8)
  (h2 : y + 1/x = 7/12)
  (h3 : x + y = 7) :
  y = 49/103 := by
  sorry

end y_value_proof_l2665_266598


namespace bus_stop_time_l2665_266577

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 64 → speed_with_stops = 48 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 15 := by
  sorry

end bus_stop_time_l2665_266577


namespace larger_number_problem_l2665_266575

theorem larger_number_problem (x y : ℕ) : 
  x * y = 40 → x + y = 13 → max x y = 8 := by sorry

end larger_number_problem_l2665_266575


namespace max_markers_is_16_l2665_266522

-- Define the prices and quantities for each option
def single_marker_price : ℕ := 2
def pack4_price : ℕ := 6
def pack8_price : ℕ := 10
def pack4_quantity : ℕ := 4
def pack8_quantity : ℕ := 8

-- Define Lisa's budget
def budget : ℕ := 20

-- Define a function to calculate the number of markers for a given combination of purchases
def markers_bought (singles pack4s pack8s : ℕ) : ℕ :=
  singles + pack4s * pack4_quantity + pack8s * pack8_quantity

-- Define a function to calculate the total cost of a combination of purchases
def total_cost (singles pack4s pack8s : ℕ) : ℕ :=
  singles * single_marker_price + pack4s * pack4_price + pack8s * pack8_price

-- Theorem: The maximum number of markers that can be bought with the given budget is 16
theorem max_markers_is_16 :
  ∀ (singles pack4s pack8s : ℕ),
    total_cost singles pack4s pack8s ≤ budget →
    markers_bought singles pack4s pack8s ≤ 16 :=
by sorry

end max_markers_is_16_l2665_266522


namespace pickle_problem_l2665_266531

/-- Pickle problem theorem -/
theorem pickle_problem (jars : ℕ) (cucumbers : ℕ) (initial_vinegar : ℕ) 
  (pickles_per_cucumber : ℕ) (vinegar_per_jar : ℕ) (remaining_vinegar : ℕ) :
  jars = 4 →
  cucumbers = 10 →
  initial_vinegar = 100 →
  pickles_per_cucumber = 6 →
  vinegar_per_jar = 10 →
  remaining_vinegar = 60 →
  (initial_vinegar - remaining_vinegar) / vinegar_per_jar = jars →
  (cucumbers * pickles_per_cucumber) / jars = 15 :=
by sorry

end pickle_problem_l2665_266531


namespace a_positive_sufficient_not_necessary_for_abs_a_positive_l2665_266578

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, a > 0 → |a| > 0) ∧
  (∃ a : ℝ, |a| > 0 ∧ ¬(a > 0)) := by
  sorry

end a_positive_sufficient_not_necessary_for_abs_a_positive_l2665_266578


namespace climbing_solution_l2665_266584

/-- Represents the climbing problem with given conditions -/
def ClimbingProblem (v : ℝ) : Prop :=
  let t₁ : ℝ := 14 / 2 + 1  -- Time on first day
  let t₂ : ℝ := 14 / 2 - 1  -- Time on second day
  let v₁ : ℝ := v - 0.5     -- Speed on first day
  let v₂ : ℝ := v           -- Speed on second day
  (v₁ * t₁ + v₂ * t₂ = 52) ∧ (t₁ + t₂ = 14)

/-- The theorem stating the solution to the climbing problem -/
theorem climbing_solution : ∃ v : ℝ, ClimbingProblem v ∧ v = 4 := by
  sorry

end climbing_solution_l2665_266584


namespace probability_spade_or_diamond_l2665_266558

theorem probability_spade_or_diamond (total_cards : ℕ) (ranks : ℕ) (suits : ℕ) 
  (h1 : total_cards = 52)
  (h2 : ranks = 13)
  (h3 : suits = 4)
  (h4 : total_cards = ranks * suits) :
  (2 : ℚ) * (ranks : ℚ) / (total_cards : ℚ) = 1 / 2 := by
  sorry

end probability_spade_or_diamond_l2665_266558


namespace smallest_four_digit_divisible_by_9_with_specific_digits_l2665_266510

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def has_odd_units_and_thousands (n : ℕ) : Prop :=
  n % 2 = 1 ∧ (n / 1000) % 2 = 1

def has_even_tens_and_hundreds (n : ℕ) : Prop :=
  ((n / 10) % 10) % 2 = 0 ∧ ((n / 100) % 10) % 2 = 0

theorem smallest_four_digit_divisible_by_9_with_specific_digits : 
  ∀ n : ℕ, is_four_digit n → 
  is_divisible_by_9 n → 
  has_odd_units_and_thousands n → 
  has_even_tens_and_hundreds n → 
  3609 ≤ n :=
sorry

end smallest_four_digit_divisible_by_9_with_specific_digits_l2665_266510
