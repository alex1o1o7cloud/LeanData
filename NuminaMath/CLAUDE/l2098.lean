import Mathlib

namespace factorial_ratio_simplification_l2098_209845

theorem factorial_ratio_simplification (N : ℕ) (h : N ≥ 2) :
  (Nat.factorial (N - 2) * (N - 1) * N) / Nat.factorial (N + 2) = 1 / ((N + 2) * (N + 1)) :=
by sorry

end factorial_ratio_simplification_l2098_209845


namespace solution_set_m_2_range_of_m_l2098_209882

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| + |2 * x + m|

-- Theorem 1: Solution set for f(x) ≤ 3 when m = 2
theorem solution_set_m_2 :
  {x : ℝ | f x 2 ≤ 3} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 0} := by sorry

-- Theorem 2: Range of m values for f(x) ≤ |2x - 3| with x ∈ [0, 1]
theorem range_of_m :
  {m : ℝ | ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x m ≤ |2 * x - 3|} = {m : ℝ | -3 ≤ m ∧ m ≤ 2} := by sorry

end solution_set_m_2_range_of_m_l2098_209882


namespace intersection_points_theorem_l2098_209849

def f (x : ℝ) : ℝ := (x - 2) * (x - 1) * (x + 1)

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (-x)

def c : ℕ := 3

def d : ℕ := 2

theorem intersection_points_theorem : 10 * c + d = 32 := by
  sorry

end intersection_points_theorem_l2098_209849


namespace capri_sun_cost_per_pouch_l2098_209897

theorem capri_sun_cost_per_pouch :
  let boxes : ℕ := 10
  let pouches_per_box : ℕ := 6
  let total_cost_dollars : ℕ := 12
  let total_pouches : ℕ := boxes * pouches_per_box
  let total_cost_cents : ℕ := total_cost_dollars * 100
  total_cost_cents / total_pouches = 20 := by sorry

end capri_sun_cost_per_pouch_l2098_209897


namespace revenue_maximized_at_optimal_price_l2098_209841

/-- Revenue function for gadget sales -/
def R (p : ℝ) : ℝ := p * (200 - 4 * p)

/-- The price that maximizes revenue -/
def optimal_price : ℝ := 25

theorem revenue_maximized_at_optimal_price :
  ∀ p : ℝ, p ≤ 40 → R p ≤ R optimal_price := by sorry

end revenue_maximized_at_optimal_price_l2098_209841


namespace arithmetic_sequence_property_l2098_209838

def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1) * d

theorem arithmetic_sequence_property :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (a 3 = 5) →
  (∀ n : ℕ, a n = arithmeticSequence (a 1) ((a 3 - a 1) / 2) n) →
  2 * (a 9) - (a 10) = 15 := by
  sorry

end arithmetic_sequence_property_l2098_209838


namespace cubic_equation_solutions_mean_l2098_209818

theorem cubic_equation_solutions_mean (x : ℝ) : 
  (x^3 + 5*x^2 - 14*x = 0) → 
  (∃ s : Finset ℝ, (∀ y ∈ s, y^3 + 5*y^2 - 14*y = 0) ∧ 
                   (∀ z, z^3 + 5*z^2 - 14*z = 0 → z ∈ s) ∧
                   (Finset.sum s id / s.card = -5/3)) := by
  sorry

end cubic_equation_solutions_mean_l2098_209818


namespace ratio_a5_b5_l2098_209876

/-- Given two arithmetic sequences a and b, S_n and T_n represent the sum of their first n terms respectively. -/
def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ := sorry

def T_n (b : ℕ → ℚ) (n : ℕ) : ℚ := sorry

/-- The ratio of S_n to T_n is equal to 7n / (n+3) for all n. -/
axiom ratio_condition {a b : ℕ → ℚ} (n : ℕ) : S_n a n / T_n b n = (7 * n) / (n + 3)

/-- The main theorem: given the ratio condition, the ratio of a_5 to b_5 is 21/4. -/
theorem ratio_a5_b5 {a b : ℕ → ℚ} : a 5 / b 5 = 21 / 4 := sorry

end ratio_a5_b5_l2098_209876


namespace function_derivative_problem_l2098_209819

theorem function_derivative_problem (a : ℝ) :
  (∀ x, f x = (2 * x + a) ^ 2) →
  (deriv f) 2 = 20 →
  a = 1 := by sorry

end function_derivative_problem_l2098_209819


namespace area_ratio_theorem_l2098_209806

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (XY : Real)
  (YZ : Real)
  (XZ : Real)

-- Define points P and Q
structure Points (t : Triangle) :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (XP : Real)
  (XQ : Real)

-- Define the area ratio
def areaRatio (t : Triangle) (pts : Points t) : ℚ := sorry

-- State the theorem
theorem area_ratio_theorem (t : Triangle) (pts : Points t) :
  t.XY = 30 →
  t.YZ = 45 →
  t.XZ = 54 →
  pts.XP = 18 →
  pts.XQ = 36 →
  areaRatio t pts = 27 / 50 := by sorry

end area_ratio_theorem_l2098_209806


namespace successful_hatch_percentage_l2098_209888

/-- The number of eggs laid by each turtle -/
def eggs_per_turtle : ℕ := 20

/-- The number of turtles -/
def num_turtles : ℕ := 6

/-- The number of hatchlings produced -/
def num_hatchlings : ℕ := 48

/-- The percentage of eggs that successfully hatch -/
def hatch_percentage : ℚ := 40

theorem successful_hatch_percentage :
  (eggs_per_turtle * num_turtles : ℚ) * (hatch_percentage / 100) = num_hatchlings :=
sorry

end successful_hatch_percentage_l2098_209888


namespace special_numbers_theorem_l2098_209846

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

def replace_greatest_with_one (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  let max_digit := max d1 (max d2 d3)
  if d1 = max_digit then
    100 + d2 * 10 + d3
  else if d2 = max_digit then
    d1 * 100 + 10 + d3
  else
    d1 * 100 + d2 * 10 + 1

theorem special_numbers_theorem :
  {n : ℕ | is_three_digit n ∧ 
           has_distinct_digits n ∧ 
           (replace_greatest_with_one n) % 30 = 0} =
  {230, 320, 560, 650, 890, 980} := by sorry

end special_numbers_theorem_l2098_209846


namespace three_red_and_at_least_one_white_mutually_exclusive_and_complementary_l2098_209855

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing 3 balls -/
def Outcome := (Color × Color × Color)

/-- The sample space of all possible outcomes when drawing 3 balls from a bag with 5 red and 5 white balls -/
def SampleSpace : Set Outcome := sorry

/-- Event: Draw three red balls -/
def ThreeRedBalls (outcome : Outcome) : Prop := 
  outcome = (Color.Red, Color.Red, Color.Red)

/-- Event: Draw three balls with at least one white ball -/
def AtLeastOneWhiteBall (outcome : Outcome) : Prop := 
  outcome.1 = Color.White ∨ outcome.2.1 = Color.White ∨ outcome.2.2 = Color.White

theorem three_red_and_at_least_one_white_mutually_exclusive_and_complementary :
  (∀ outcome ∈ SampleSpace, ¬(ThreeRedBalls outcome ∧ AtLeastOneWhiteBall outcome)) ∧ 
  (∀ outcome ∈ SampleSpace, ThreeRedBalls outcome ∨ AtLeastOneWhiteBall outcome) := by
  sorry

end three_red_and_at_least_one_white_mutually_exclusive_and_complementary_l2098_209855


namespace train_length_calculation_l2098_209831

/-- The length of a train given its speed, a man's speed in the opposite direction, and the time it takes to pass the man -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (pass_time : ℝ) :
  train_speed = 60 →
  man_speed = 6 →
  pass_time = 32.99736021118311 →
  ∃ (train_length : ℝ), 
    (train_length ≥ 604.99 ∧ train_length ≤ 605.01) ∧
    train_length = (train_speed + man_speed) * (5 / 18) * pass_time :=
by sorry

end train_length_calculation_l2098_209831


namespace recliner_price_drop_l2098_209890

theorem recliner_price_drop 
  (initial_quantity : ℝ) 
  (initial_price : ℝ) 
  (quantity_increase_ratio : ℝ) 
  (revenue_increase_ratio : ℝ) 
  (h1 : quantity_increase_ratio = 1.60) 
  (h2 : revenue_increase_ratio = 1.2800000000000003) : 
  let new_quantity := initial_quantity * quantity_increase_ratio
  let new_price := initial_price * (revenue_increase_ratio / quantity_increase_ratio)
  new_price / initial_price = 0.80 := by
sorry

end recliner_price_drop_l2098_209890


namespace unit_digit_product_l2098_209801

theorem unit_digit_product : (3^68 * 6^59 * 7^71) % 10 = 8 := by
  sorry

end unit_digit_product_l2098_209801


namespace point_not_on_transformed_plane_l2098_209822

/-- The original plane equation -/
def plane_a (x y z : ℝ) : Prop := 2*x + 3*y + z - 1 = 0

/-- The similarity transformation with scale factor k -/
def similarity_transform (k : ℝ) (x y z : ℝ) : Prop := 2*x + 3*y + z - k = 0

/-- The point A -/
def point_A : ℝ × ℝ × ℝ := (1, 2, -1)

/-- The scale factor -/
def k : ℝ := 2

/-- Theorem stating that point A does not lie on the transformed plane -/
theorem point_not_on_transformed_plane : 
  ¬ similarity_transform k point_A.1 point_A.2.1 point_A.2.2 :=
sorry

end point_not_on_transformed_plane_l2098_209822


namespace dinner_cost_is_120_l2098_209881

/-- Calculates the cost of dinner before tip given the total cost, ticket price, number of tickets, limo hourly rate, limo hours, and tip percentage. -/
def dinner_cost (total : ℚ) (ticket_price : ℚ) (num_tickets : ℕ) (limo_rate : ℚ) (limo_hours : ℕ) (tip_percent : ℚ) : ℚ :=
  let ticket_cost := ticket_price * num_tickets
  let limo_cost := limo_rate * limo_hours
  let dinner_with_tip := total - (ticket_cost + limo_cost)
  dinner_with_tip / (1 + tip_percent)

/-- Proves that the cost of dinner before tip is $120 given the specified conditions. -/
theorem dinner_cost_is_120 :
  dinner_cost 836 100 2 80 6 (30/100) = 120 := by
  sorry

end dinner_cost_is_120_l2098_209881


namespace compound_molecular_weight_l2098_209878

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (Ca I C H : ℕ) (wCa wI wC wH : ℝ) : ℝ :=
  Ca * wCa + I * wI + C * wC + H * wH

/-- The molecular weight of the given compound is 602.794 amu -/
theorem compound_molecular_weight :
  let Ca : ℕ := 2
  let I : ℕ := 4
  let C : ℕ := 1
  let H : ℕ := 3
  let wCa : ℝ := 40.08
  let wI : ℝ := 126.90
  let wC : ℝ := 12.01
  let wH : ℝ := 1.008
  molecularWeight Ca I C H wCa wI wC wH = 602.794 := by
  sorry

end compound_molecular_weight_l2098_209878


namespace rectangle_Q_coordinates_l2098_209863

/-- A rectangle in a 2D plane --/
structure Rectangle where
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The specific rectangle from the problem --/
def problemRectangle : Rectangle where
  O := (0, 0)
  P := (0, 3)
  R := (5, 0)
  Q := (5, 3)  -- We'll prove this is correct

/-- Predicate to check if four points form a rectangle --/
def isRectangle (rect : Rectangle) : Prop :=
  -- Opposite sides are parallel and equal in length
  (rect.O.1 = rect.P.1 ∧ rect.Q.1 = rect.R.1) ∧
  (rect.O.2 = rect.R.2 ∧ rect.P.2 = rect.Q.2) ∧
  (rect.P.1 - rect.O.1)^2 + (rect.P.2 - rect.O.2)^2 =
  (rect.Q.1 - rect.R.1)^2 + (rect.Q.2 - rect.R.2)^2 ∧
  (rect.R.1 - rect.O.1)^2 + (rect.R.2 - rect.O.2)^2 =
  (rect.Q.1 - rect.P.1)^2 + (rect.Q.2 - rect.P.2)^2

theorem rectangle_Q_coordinates :
  isRectangle problemRectangle →
  problemRectangle.Q = (5, 3) := by
  sorry

end rectangle_Q_coordinates_l2098_209863


namespace fenced_area_with_cutouts_l2098_209887

theorem fenced_area_with_cutouts (yard_length yard_width cutout1_side cutout2_side : ℝ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 15)
  (h3 : cutout1_side = 4)
  (h4 : cutout2_side = 2) :
  yard_length * yard_width - (cutout1_side * cutout1_side + cutout2_side * cutout2_side) = 280 := by
  sorry

end fenced_area_with_cutouts_l2098_209887


namespace product_mod_seven_l2098_209842

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end product_mod_seven_l2098_209842


namespace max_white_rooks_8x8_l2098_209862

/-- Represents a chessboard configuration with black and white rooks -/
structure ChessboardConfig where
  size : Nat
  blackRooks : Nat
  whiteRooks : Nat
  differentCells : Bool
  onlyAttackOpposite : Bool

/-- Defines the maximum number of white rooks for a given configuration -/
def maxWhiteRooks (config : ChessboardConfig) : Nat :=
  sorry

/-- Theorem stating the maximum number of white rooks for the given configuration -/
theorem max_white_rooks_8x8 :
  let config : ChessboardConfig := {
    size := 8,
    blackRooks := 6,
    whiteRooks := 14,
    differentCells := true,
    onlyAttackOpposite := true
  }
  maxWhiteRooks config = 14 := by sorry

end max_white_rooks_8x8_l2098_209862


namespace x_plus_y_values_l2098_209829

theorem x_plus_y_values (x y : ℝ) (hx : |x| = 3) (hy : |y| = 6) (hxy : x > y) :
  (x + y = -3 ∨ x + y = -9) ∧ ∀ z, (x + y = z → z = -3 ∨ z = -9) :=
by sorry

end x_plus_y_values_l2098_209829


namespace aprils_roses_l2098_209844

theorem aprils_roses (initial_roses : ℕ) 
  (rose_price : ℕ) 
  (total_earnings : ℕ) 
  (roses_left : ℕ) : 
  rose_price = 4 → 
  total_earnings = 36 → 
  roses_left = 4 → 
  initial_roses = 13 := by
  sorry

end aprils_roses_l2098_209844


namespace exists_n_congruence_l2098_209874

/-- ν(n) denotes the exponent of 2 in the prime factorization of n! -/
def ν (n : ℕ) : ℕ := sorry

/-- For any positive integers a and m, there exists an integer n > 1 such that ν(n) ≡ a (mod m) -/
theorem exists_n_congruence (a m : ℕ+) : ∃ n : ℕ, n > 1 ∧ ν n % m = a % m := by
  sorry

end exists_n_congruence_l2098_209874


namespace inequalities_on_positive_reals_l2098_209811

theorem inequalities_on_positive_reals :
  ∀ x : ℝ, x > 0 →
    (Real.log x < x) ∧
    (Real.sin x < x) ∧
    (Real.exp x > x) := by
  sorry

end inequalities_on_positive_reals_l2098_209811


namespace unique_solution_when_k_zero_l2098_209889

/-- The equation has exactly one solution when k = 0 -/
theorem unique_solution_when_k_zero :
  ∃! x : ℝ, (x + 2) / (0 * x - 1) = x :=
sorry

end unique_solution_when_k_zero_l2098_209889


namespace P_inter_Q_l2098_209898

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem P_inter_Q : P ∩ Q = {1, 2, 3} := by sorry

end P_inter_Q_l2098_209898


namespace geometric_sequence_sum_l2098_209837

/-- Given a geometric sequence {a_n}, prove that if a_1 + a_2 = 40 and a_3 + a_4 = 60, then a_7 + a_8 = 135 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_sum1 : a 1 + a 2 = 40) (h_sum2 : a 3 + a 4 = 60) : a 7 + a 8 = 135 := by
  sorry

end geometric_sequence_sum_l2098_209837


namespace hexagon_extended_point_distance_l2098_209810

/-- Regular hexagon with side length 1 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (is_regular : ∀ (X Y : ℝ × ℝ), (X, Y) ∈ [(A, B), (B, C), (C, D), (D, E), (E, F), (F, A)] → dist X Y = 1)

/-- Point Y extended from A such that BY = 4AB -/
def extend_point (h : RegularHexagon) (Y : ℝ × ℝ) : Prop :=
  dist h.B Y = 4 * dist h.A h.B

/-- The length of segment EY is √21 -/
theorem hexagon_extended_point_distance (h : RegularHexagon) (Y : ℝ × ℝ) 
  (h_extend : extend_point h Y) : 
  dist h.E Y = Real.sqrt 21 := by sorry

end hexagon_extended_point_distance_l2098_209810


namespace even_function_implies_a_zero_l2098_209873

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 4

-- State the theorem
theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 :=
by sorry

end even_function_implies_a_zero_l2098_209873


namespace center_is_eight_l2098_209804

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions are adjacent --/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 + 1 = q.2)) ∨
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 + 1 = q.1))

/-- Check if the grid satisfies the consecutive adjacency property --/
def consecutive_adjacent (g : Grid) : Prop :=
  ∀ n : Fin 8, ∃ p q : Fin 3 × Fin 3, 
    g p.1 p.2 = n ∧ g q.1 q.2 = n + 1 ∧ adjacent p q

/-- Sum of corner numbers in the grid --/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- Sum of numbers in the middle column --/
def middle_column_sum (g : Grid) : Nat :=
  g 0 1 + g 1 1 + g 2 1

theorem center_is_eight (g : Grid) 
  (h1 : ∀ n : Fin 9, ∃! p : Fin 3 × Fin 3, g p.1 p.2 = n)
  (h2 : consecutive_adjacent g)
  (h3 : corner_sum g = 20)
  (h4 : Even (middle_column_sum g)) :
  g 1 1 = 8 := by
  sorry

end center_is_eight_l2098_209804


namespace school_teachers_count_l2098_209865

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total = 2400)
  (h2 : sample_size = 150)
  (h3 : students_in_sample = 135) :
  total - (total * students_in_sample / sample_size) = 240 :=
by sorry

end school_teachers_count_l2098_209865


namespace factor_x12_minus_1024_l2098_209870

theorem factor_x12_minus_1024 (x : ℝ) : 
  x^12 - 1024 = (x^6 + 32) * (x^3 + 4 * Real.sqrt 2) * (x^3 - 4 * Real.sqrt 2) := by
  sorry

end factor_x12_minus_1024_l2098_209870


namespace rotation_exists_l2098_209892

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line
  mk :: -- Constructor

/-- Represents a point in 3D space -/
structure Point3D where
  -- Add necessary fields for a 3D point
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane
  mk :: -- Constructor

/-- Represents a rotation in 3D space -/
structure Rotation3D where
  -- Add necessary fields for a 3D rotation
  mk :: -- Constructor
  apply : Point3D → Point3D  -- Applies the rotation to a point

def are_skew (l1 l2 : Line3D) : Prop := sorry

def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

def rotation_maps (r : Rotation3D) (l1 l2 : Line3D) (p1 p2 : Point3D) : Prop := sorry

def plane_of_symmetry (p1 p2 : Point3D) : Plane3D := sorry

def plane_intersection (p1 p2 : Plane3D) : Line3D := sorry

theorem rotation_exists (a a' : Line3D) (A : Point3D) (A' : Point3D) 
  (h1 : are_skew a a')
  (h2 : point_on_line A a)
  (h3 : point_on_line A' a') :
  ∃ (l : Line3D), ∃ (r : Rotation3D), rotation_maps r a a' A A' := by
  sorry

end rotation_exists_l2098_209892


namespace total_amount_is_sum_of_shares_l2098_209816

/-- Represents the time in days it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℕ

/-- Represents the share of money received by a person -/
structure Share where
  amount : ℕ

/-- Represents a worker with their individual work time and share -/
structure Worker where
  workTime : WorkTime
  share : Share

/-- Theorem: The total amount received for the work is the sum of individual shares -/
theorem total_amount_is_sum_of_shares 
  (a b c : Worker)
  (h1 : a.workTime.days = 6)
  (h2 : b.workTime.days = 8)
  (h3 : a.share.amount = 300)
  (h4 : b.share.amount = 225)
  (h5 : c.share.amount = 75) :
  a.share.amount + b.share.amount + c.share.amount = 600 := by
  sorry

end total_amount_is_sum_of_shares_l2098_209816


namespace travel_ratio_l2098_209868

theorem travel_ratio (george joseph patrick zack : ℕ) : 
  george = 6 →
  joseph = george / 2 →
  patrick = joseph * 3 →
  zack = 18 →
  zack / patrick = 2 :=
by sorry

end travel_ratio_l2098_209868


namespace friend_distribution_problem_l2098_209879

/-- The number of friends that satisfies the given conditions --/
def num_friends : ℕ := 16

/-- The total amount distributed in rupees --/
def total_amount : ℕ := 5000

/-- The decrease in amount per person if there were 8 more friends --/
def decrease_amount : ℕ := 125

theorem friend_distribution_problem :
  (total_amount / num_friends : ℚ) - (total_amount / (num_friends + 8) : ℚ) = decrease_amount ∧
  num_friends > 0 := by
  sorry

#check friend_distribution_problem

end friend_distribution_problem_l2098_209879


namespace banana_cost_l2098_209820

theorem banana_cost (num_bananas : ℕ) (num_oranges : ℕ) (orange_cost : ℚ) (total_cost : ℚ) :
  num_bananas = 5 →
  num_oranges = 10 →
  orange_cost = 3/2 →
  total_cost = 25 →
  (total_cost - num_oranges * orange_cost) / num_bananas = 2 :=
by sorry

end banana_cost_l2098_209820


namespace tommy_steaks_l2098_209883

/-- The number of steaks needed for a family dinner -/
def steaks_needed (family_members : ℕ) (pounds_per_member : ℕ) (ounces_per_steak : ℕ) : ℕ :=
  let total_ounces := family_members * pounds_per_member * 16
  (total_ounces + ounces_per_steak - 1) / ounces_per_steak

/-- Theorem: Tommy needs to buy 4 steaks for his family -/
theorem tommy_steaks : steaks_needed 5 1 20 = 4 := by
  sorry

end tommy_steaks_l2098_209883


namespace p_sufficient_not_necessary_for_q_l2098_209899

/-- Proposition p -/
def p (x : ℝ) : Prop := x^2 - x - 20 > 0

/-- Proposition q -/
def q (x : ℝ) : Prop := |x| - 2 > 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_for_q_l2098_209899


namespace union_of_M_and_N_l2098_209803

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 2) ≤ 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end union_of_M_and_N_l2098_209803


namespace unique_zero_point_b_range_l2098_209817

-- Define the function f_n
def f_n (n : ℕ) (b c : ℝ) (x : ℝ) : ℝ := x^n + b*x + c

-- Part I
theorem unique_zero_point (n : ℕ) (h : n ≥ 2) :
  ∃! x, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ f_n n 1 (-1) x = 0 :=
sorry

-- Part II
theorem b_range (h : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1 : ℝ) 1 → x₂ ∈ Set.Icc (-1 : ℝ) 1 →
  |f_n 2 b c x₁ - f_n 2 b c x₂| ≤ 4) :
  b ∈ Set.Icc (-2 : ℝ) 2 :=
sorry

end unique_zero_point_b_range_l2098_209817


namespace expression_evaluation_l2098_209867

theorem expression_evaluation : 
  let x : ℝ := 2
  (2 * x + 3) * (2 * x - 3) + (x - 2)^2 - 3 * x * (x - 1) = 1 := by
  sorry

end expression_evaluation_l2098_209867


namespace entropy_increase_l2098_209834

-- Define the temperature in Kelvin
def T : ℝ := 298

-- Define the enthalpy change in kJ/mol
def ΔH : ℝ := 2171

-- Define the entropy change in J/(mol·K)
def ΔS : ℝ := 635.5

-- Theorem to prove that the entropy change is positive
theorem entropy_increase : ΔS > 0 := by
  sorry

end entropy_increase_l2098_209834


namespace remainder_3_180_mod_5_l2098_209853

theorem remainder_3_180_mod_5 : 3^180 % 5 = 1 := by
  sorry

end remainder_3_180_mod_5_l2098_209853


namespace expand_and_factor_l2098_209866

theorem expand_and_factor (a b c : ℝ) : (a + b - c) * (a - b + c) = (a + (b - c)) * (a - (b - c)) := by
  sorry

end expand_and_factor_l2098_209866


namespace sine_cosine_roots_l2098_209891

theorem sine_cosine_roots (θ : Real) (k : Real) 
  (h1 : θ > 0 ∧ θ < 2 * Real.pi)
  (h2 : (Real.sin θ)^2 - k * (Real.sin θ) + k + 1 = 0)
  (h3 : (Real.cos θ)^2 - k * (Real.cos θ) + k + 1 = 0) :
  k = -1 := by
sorry

end sine_cosine_roots_l2098_209891


namespace sum_of_xyz_l2098_209821

theorem sum_of_xyz (x y z : ℤ) 
  (hz : z = 4)
  (hxy : x + y = 7)
  (hxz : x + z = 8) : 
  x + y + z = 11 := by
sorry

end sum_of_xyz_l2098_209821


namespace angle_A_value_l2098_209805

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- State the theorem
theorem angle_A_value (abc : Triangle) (h : abc.b = 2 * abc.a * Real.sin abc.B) :
  abc.A = π/6 ∨ abc.A = 5*π/6 := by
  sorry

end angle_A_value_l2098_209805


namespace cube_surface_area_l2098_209832

theorem cube_surface_area (V : ℝ) (h : V = 64) : 
  6 * (V ^ (1/3))^2 = 96 := by
  sorry

end cube_surface_area_l2098_209832


namespace russian_doll_purchase_l2098_209852

/-- Given a person's savings for a certain number of items at an original price,
    calculate how many items they can buy when the price drops to a new lower price. -/
theorem russian_doll_purchase (original_price new_price : ℚ) (original_quantity : ℕ) :
  original_price > 0 →
  new_price > 0 →
  new_price < original_price →
  (original_price * original_quantity) / new_price = 20 :=
by
  sorry

#check russian_doll_purchase (4 : ℚ) (3 : ℚ) 15

end russian_doll_purchase_l2098_209852


namespace unique_solution_lcm_gcd_l2098_209895

theorem unique_solution_lcm_gcd : 
  ∃! n : ℕ+, n.lcm 120 = n.gcd 120 + 300 ∧ n = 180 := by sorry

end unique_solution_lcm_gcd_l2098_209895


namespace tiffany_green_buckets_l2098_209826

/-- Carnival ring toss game -/
structure CarnivalGame where
  total_money : ℕ
  cost_per_play : ℕ
  rings_per_play : ℕ
  red_bucket_points : ℕ
  green_bucket_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  total_points : ℕ

/-- Calculate the number of green buckets hit -/
def green_buckets_hit (game : CarnivalGame) : ℕ :=
  (game.total_points - game.red_buckets_hit * game.red_bucket_points) / game.green_bucket_points

/-- Theorem: Tiffany hit 10 green buckets -/
theorem tiffany_green_buckets :
  let game : CarnivalGame := {
    total_money := 3,
    cost_per_play := 1,
    rings_per_play := 5,
    red_bucket_points := 2,
    green_bucket_points := 3,
    games_played := 2,
    red_buckets_hit := 4,
    total_points := 38
  }
  green_buckets_hit game = 10 := by
  sorry

end tiffany_green_buckets_l2098_209826


namespace quadratic_inequality_theorem_l2098_209856

/-- The quadratic inequality function -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

/-- The solution set when m = 0 -/
def solution_set_m_zero : Set ℝ := {x | -2 < x ∧ x < 1}

/-- The range of m for which the solution set is ℝ -/
def m_range : Set ℝ := {m | 1 ≤ m ∧ m < 9}

theorem quadratic_inequality_theorem :
  (∀ x, x ∈ solution_set_m_zero ↔ f 0 x > 0) ∧
  (∀ m, (∀ x, f m x > 0) ↔ m ∈ m_range) :=
sorry

end quadratic_inequality_theorem_l2098_209856


namespace baron_munchausen_contradiction_l2098_209813

-- Define the total distance and time of the walk
variable (S : ℝ) -- Total distance
variable (T : ℝ) -- Total time

-- Define the speeds
def speed1 : ℝ := 5 -- Speed for half the distance
def speed2 : ℝ := 6 -- Speed for half the time

-- Theorem: It's impossible to satisfy both conditions
theorem baron_munchausen_contradiction :
  ¬(∃ (S T : ℝ), S > 0 ∧ T > 0 ∧
    (S / 2) / speed1 + (S / 2) / speed2 = T ∧
    (S / 2) + speed2 * (T / 2) = S) :=
sorry

end baron_munchausen_contradiction_l2098_209813


namespace dihedral_angle_is_120_degrees_l2098_209835

/-- A regular tetrahedron with a circumscribed sphere -/
structure RegularTetrahedronWithSphere where
  /-- The height of the tetrahedron -/
  height : ℝ
  /-- The diameter of the circumscribed sphere -/
  sphere_diameter : ℝ
  /-- The diameter of the sphere is 9 times the height of the tetrahedron -/
  sphere_diameter_relation : sphere_diameter = 9 * height

/-- The dihedral angle between two lateral faces of a regular tetrahedron -/
def dihedral_angle (t : RegularTetrahedronWithSphere) : ℝ :=
  sorry

/-- Theorem: The dihedral angle between two lateral faces of a regular tetrahedron
    with the given sphere relation is 120 degrees -/
theorem dihedral_angle_is_120_degrees (t : RegularTetrahedronWithSphere) :
  dihedral_angle t = 120 * π / 180 :=
sorry

end dihedral_angle_is_120_degrees_l2098_209835


namespace quadratic_equations_solutions_l2098_209827

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 5*x + 4 = 0 ↔ x = 4 ∨ x = 1) ∧
  (∃ x : ℝ, x^2 = 4 - 2*x ↔ x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5) :=
sorry

end quadratic_equations_solutions_l2098_209827


namespace imaginary_part_of_z_l2098_209828

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z := i^2018 / (i^2019 - 1)
  Complex.im z = -1/2 := by sorry

end imaginary_part_of_z_l2098_209828


namespace income_analysis_l2098_209812

/-- Represents the income status of a household -/
inductive IncomeStatus
| Above10000
| Below10000

/-- Represents a region with households -/
structure Region where
  totalHouseholds : ℕ
  aboveThreshold : ℕ

/-- Represents the sample data -/
structure SampleData where
  regionA : Region
  regionB : Region
  totalSample : ℕ

/-- The probability of selecting a household with income above 10000 from a region -/
def probAbove10000 (r : Region) : ℚ :=
  r.aboveThreshold / r.totalHouseholds

/-- The expected value of X (number of households with income > 10000 when selecting one from each region) -/
def expectedX (sd : SampleData) : ℚ :=
  (probAbove10000 sd.regionA + probAbove10000 sd.regionB) / 2

/-- The main theorem to be proved -/
theorem income_analysis (sd : SampleData)
  (h1 : sd.regionA.totalHouseholds = 300)
  (h2 : sd.regionA.aboveThreshold = 100)
  (h3 : sd.regionB.totalHouseholds = 200)
  (h4 : sd.regionB.aboveThreshold = 150)
  (h5 : sd.totalSample = 500) :
  probAbove10000 sd.regionA = 1/3 ∧ expectedX sd = 13/12 := by
  sorry

end income_analysis_l2098_209812


namespace profit_share_calculation_l2098_209860

theorem profit_share_calculation (investment_A investment_B investment_C : ℕ)
  (profit_difference_AC : ℕ) (profit_share_B : ℕ) :
  investment_A = 6000 →
  investment_B = 8000 →
  investment_C = 10000 →
  profit_difference_AC = 500 →
  profit_share_B = 1000 :=
by sorry

end profit_share_calculation_l2098_209860


namespace evaluate_expression_l2098_209893

theorem evaluate_expression :
  -(16 / 4 * 11 - 70 + 5 * 11) = -29 := by
  sorry

end evaluate_expression_l2098_209893


namespace rhombus_side_length_l2098_209808

/-- A rhombus with perimeter 32 has side length 8 -/
theorem rhombus_side_length (perimeter : ℝ) (h1 : perimeter = 32) : perimeter / 4 = 8 := by
  sorry

end rhombus_side_length_l2098_209808


namespace quilt_transformation_l2098_209851

/-- Given a rectangular quilt with width 6 feet and an unknown length, and a square quilt with side length 12 feet, 
    if their areas are equal, then the length of the rectangular quilt is 24 feet. -/
theorem quilt_transformation (length : ℝ) : 
  (6 * length = 12 * 12) → length = 24 := by
  sorry

end quilt_transformation_l2098_209851


namespace H_surjective_l2098_209840

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2| + x

-- State the theorem
theorem H_surjective : Function.Surjective H := by sorry

end H_surjective_l2098_209840


namespace series_equals_ten_implies_k_equals_sixteen_l2098_209825

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The series in question -/
noncomputable def series (k : ℝ) : ℝ := 
  4 + geometric_sum ((4 + k) / 5) (1 / 5)

theorem series_equals_ten_implies_k_equals_sixteen :
  ∃ k : ℝ, series k = 10 ∧ k = 16 := by sorry

end series_equals_ten_implies_k_equals_sixteen_l2098_209825


namespace symmetric_point_of_2_5_l2098_209814

/-- Given a point P(a,b) and a line with equation x+y=0, 
    the symmetric point Q(x,y) satisfies:
    1. x + y = 0 (lies on the line)
    2. The midpoint of PQ lies on the line
    3. PQ is perpendicular to the line -/
def is_symmetric_point (a b x y : ℝ) : Prop :=
  x + y = 0 ∧
  (a + x) / 2 + (b + y) / 2 = 0 ∧
  (x - a) = (b - y)

/-- The point symmetric to P(2,5) with respect to the line x+y=0 
    has coordinates (-5,-2) -/
theorem symmetric_point_of_2_5 : 
  is_symmetric_point 2 5 (-5) (-2) := by sorry

end symmetric_point_of_2_5_l2098_209814


namespace binomial_20_19_l2098_209823

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_20_19_l2098_209823


namespace range_of_m_plus_n_l2098_209824

noncomputable def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, f m n x = 0) ∧ 
  (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
sorry

end range_of_m_plus_n_l2098_209824


namespace cooking_time_calculation_l2098_209864

/-- Represents the cooking time for each food item -/
structure CookingTime where
  waffles : ℕ
  steak : ℕ
  chili : ℕ
  fries : ℕ

/-- Represents the quantity of each food item to be cooked -/
structure CookingQuantity where
  waffles : ℕ
  steak : ℕ
  chili : ℕ
  fries : ℕ

/-- Calculates the total cooking time given the cooking times and quantities -/
def totalCookingTime (time : CookingTime) (quantity : CookingQuantity) : ℕ :=
  time.waffles * quantity.waffles +
  time.steak * quantity.steak +
  time.chili * quantity.chili +
  time.fries * quantity.fries

/-- Theorem: Given the specified cooking times and quantities, the total cooking time is 218 minutes -/
theorem cooking_time_calculation (time : CookingTime) (quantity : CookingQuantity)
  (hw : time.waffles = 10)
  (hs : time.steak = 6)
  (hc : time.chili = 20)
  (hf : time.fries = 15)
  (qw : quantity.waffles = 5)
  (qs : quantity.steak = 8)
  (qc : quantity.chili = 3)
  (qf : quantity.fries = 4) :
  totalCookingTime time quantity = 218 := by
  sorry

end cooking_time_calculation_l2098_209864


namespace power_mod_nine_l2098_209871

theorem power_mod_nine (x : ℤ) : x = 5 → x^46655 % 9 = 5 := by
  sorry

end power_mod_nine_l2098_209871


namespace point_C_coordinates_l2098_209894

-- Define the points and vectors
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)

-- Define vector AB
def vecAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector AC in terms of AB
def vecAC : ℝ × ℝ := (2 * vecAB.1, 2 * vecAB.2)

-- Define point C
def C : ℝ × ℝ := (A.1 + vecAC.1, A.2 + vecAC.2)

-- Theorem statement
theorem point_C_coordinates : C = (-3, 9) := by
  sorry

end point_C_coordinates_l2098_209894


namespace team_selection_count_l2098_209843

def total_students : ℕ := 11
def num_girls : ℕ := 3
def num_boys : ℕ := 8
def team_size : ℕ := 5

theorem team_selection_count :
  (Nat.choose total_students team_size) - (Nat.choose num_boys team_size) = 406 :=
by sorry

end team_selection_count_l2098_209843


namespace fractional_equation_solution_l2098_209884

theorem fractional_equation_solution :
  ∃ x : ℝ, (2 * x) / (x - 3) = 1 ∧ x = -3 :=
by sorry

end fractional_equation_solution_l2098_209884


namespace kids_left_playing_result_l2098_209800

/-- The number of kids left playing soccer -/
def kids_left_playing (initial : ℝ) (left : ℝ) : ℝ :=
  initial - left

/-- Theorem stating the number of kids left playing soccer -/
theorem kids_left_playing_result :
  kids_left_playing 22.5 14.3 = 8.2 := by sorry

end kids_left_playing_result_l2098_209800


namespace fifth_term_is_eight_l2098_209854

def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fibonacci_like n + fibonacci_like (n + 1)

theorem fifth_term_is_eight : fibonacci_like 4 = 8 := by
  sorry

end fifth_term_is_eight_l2098_209854


namespace copy_machines_output_l2098_209886

/-- The number of copies made by two machines in a given time -/
def total_copies (rate1 rate2 time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

/-- Theorem stating that two machines with rates 25 and 55 copies per minute
    make 2400 copies in 30 minutes -/
theorem copy_machines_output : total_copies 25 55 30 = 2400 := by
  sorry

end copy_machines_output_l2098_209886


namespace xiao_ming_math_score_l2098_209877

theorem xiao_ming_math_score :
  let average_three := 94
  let subjects := 3
  let average_two := average_three - 1
  let total_score := average_three * subjects
  let chinese_english_score := average_two * (subjects - 1)
  total_score - chinese_english_score = 96 :=
by
  sorry

end xiao_ming_math_score_l2098_209877


namespace two_digit_product_4320_l2098_209833

theorem two_digit_product_4320 :
  ∃! (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4320 ∧ a = 60 ∧ b = 72 := by
  sorry

end two_digit_product_4320_l2098_209833


namespace coordinates_of_point_b_l2098_209830

/-- Given a line segment AB with length 3, parallel to the y-axis, and point A at coordinates (-1, 2),
    the coordinates of point B must be either (-1, 5) or (-1, -1). -/
theorem coordinates_of_point_b (A B : ℝ × ℝ) : 
  A = (-1, 2) → 
  (B.1 - A.1 = 0) →  -- AB is parallel to y-axis
  ((B.1 - A.1)^2 + (B.2 - A.2)^2 = 3^2) →  -- AB length is 3
  (B = (-1, 5) ∨ B = (-1, -1)) := by
sorry

end coordinates_of_point_b_l2098_209830


namespace sum_of_distinct_integers_l2098_209850

theorem sum_of_distinct_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
  sorry

end sum_of_distinct_integers_l2098_209850


namespace last_digit_of_one_over_two_to_twenty_l2098_209880

theorem last_digit_of_one_over_two_to_twenty (n : ℕ) :
  n = 20 →
  ∃ k : ℕ, (1 : ℚ) / (2^n) = k * (1 / 10^n) + 5 * (1 / 10^n) :=
sorry

end last_digit_of_one_over_two_to_twenty_l2098_209880


namespace Q_has_35_digits_l2098_209807

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The product of two large numbers -/
def Q : ℕ := 6789432567123456789 * 98765432345678

/-- Theorem stating that Q has 35 digits -/
theorem Q_has_35_digits : num_digits Q = 35 := by sorry

end Q_has_35_digits_l2098_209807


namespace savings_calculation_l2098_209802

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 10 / 4 →
  income = 19000 →
  savings = income - expenditure →
  savings = 11400 := by
sorry

end savings_calculation_l2098_209802


namespace inverse_sum_equals_negative_twelve_l2098_209815

-- Define the function f
def f (x : ℝ) : ℝ := x * |x| + 3 * x

-- State the theorem
theorem inverse_sum_equals_negative_twelve :
  ∃ (a b : ℝ), f a = 9 ∧ f b = -121 ∧ a + b = -12 :=
sorry

end inverse_sum_equals_negative_twelve_l2098_209815


namespace classroom_students_count_l2098_209861

theorem classroom_students_count : ∃! n : ℕ, n < 60 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ n = 22 := by
  sorry

end classroom_students_count_l2098_209861


namespace union_of_A_and_B_l2098_209836

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by sorry

end union_of_A_and_B_l2098_209836


namespace expand_binomial_product_l2098_209839

theorem expand_binomial_product (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end expand_binomial_product_l2098_209839


namespace birthday_money_l2098_209857

theorem birthday_money (age : ℕ) (money : ℕ) : 
  age = 3 * 3 →
  money = 5 * age →
  money = 45 := by
sorry

end birthday_money_l2098_209857


namespace complex_fraction_equality_l2098_209859

theorem complex_fraction_equality : 1 + (1 / (1 + (1 / (1 + 1)))) = 5 / 3 := by
  sorry

end complex_fraction_equality_l2098_209859


namespace exactly_five_cheaper_points_l2098_209872

-- Define the cost function
def C (n : ℕ) : ℕ :=
  if n ≥ 50 then 13 * n
  else if n ≥ 20 then 14 * n
  else 15 * n

-- Define the property we want to prove
def cheaper_to_buy_more (n : ℕ) : Prop :=
  C (n + 1) < C n

-- Theorem statement
theorem exactly_five_cheaper_points :
  ∃ (S : Finset ℕ), S.card = 5 ∧ 
  (∀ n, n ∈ S ↔ cheaper_to_buy_more n) :=
sorry

end exactly_five_cheaper_points_l2098_209872


namespace original_list_size_l2098_209885

/-- The number of integers in the original list -/
def n : ℕ := sorry

/-- The mean of the original list -/
def m : ℚ := sorry

/-- The sum of the integers in the original list -/
def original_sum : ℚ := n * m

/-- The equation representing the first condition -/
axiom first_condition : (m + 2) * (n + 1) = original_sum + 15

/-- The equation representing the second condition -/
axiom second_condition : (m + 1) * (n + 2) = original_sum + 16

theorem original_list_size : n = 4 := by sorry

end original_list_size_l2098_209885


namespace a_explicit_formula_l2098_209848

/-- Sequence {a_n} defined recursively --/
def a : ℕ → ℚ
  | 0 => 0
  | n + 1 => a n + (n + 1)^3

/-- Theorem stating the explicit formula for a_n --/
theorem a_explicit_formula (n : ℕ) : a n = n^2 * (n + 1)^2 / 4 := by
  sorry

end a_explicit_formula_l2098_209848


namespace intersection_range_l2098_209858

-- Define the curve
def curve (x y : ℝ) : Prop :=
  Real.sqrt (1 - (y - 1)^2) = abs x - 1

-- Define the line
def line (k x y : ℝ) : Prop :=
  k * x - y = 2

-- Define the intersection condition
def intersect_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (k ≥ -2 ∧ k < -4/3) ∨ (k > 4/3 ∧ k ≤ 2)

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersect_at_two_points k ↔ k_range k :=
by sorry

end intersection_range_l2098_209858


namespace fg_difference_of_squares_l2098_209896

def f (x : ℝ) : ℝ := x - 2

def g (x : ℝ) : ℝ := 2 * x + 4

theorem fg_difference_of_squares : (f (g 3))^2 - (g (f 3))^2 = 28 := by
  sorry

end fg_difference_of_squares_l2098_209896


namespace num_al_sandwiches_l2098_209809

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 6

/-- Represents whether turkey is available. -/
def has_turkey : Prop := True

/-- Represents whether roast beef is available. -/
def has_roast_beef : Prop := True

/-- Represents whether swiss cheese is available. -/
def has_swiss_cheese : Prop := True

/-- Represents whether rye bread is available. -/
def has_rye_bread : Prop := True

/-- Represents the number of sandwich combinations with turkey and swiss cheese. -/
def turkey_swiss_combos : ℕ := num_bread

/-- Represents the number of sandwich combinations with rye bread and roast beef. -/
def rye_roast_beef_combos : ℕ := num_cheese

/-- Theorem stating the number of different sandwiches Al can order. -/
theorem num_al_sandwiches : 
  num_bread * num_meat * num_cheese - turkey_swiss_combos - rye_roast_beef_combos = 199 := by
  sorry

end num_al_sandwiches_l2098_209809


namespace percentage_for_sobel_l2098_209875

/-- Represents the percentage of voters who are male -/
def male_percentage : ℝ := 60

/-- Represents the percentage of female voters who voted for Lange -/
def female_for_lange : ℝ := 35

/-- Represents the percentage of male voters who voted for Sobel -/
def male_for_sobel : ℝ := 44

/-- Theorem stating the percentage of total voters who voted for Sobel -/
theorem percentage_for_sobel :
  let female_percentage := 100 - male_percentage
  let female_for_sobel := 100 - female_for_lange
  let total_for_sobel := (male_percentage * male_for_sobel + female_percentage * female_for_sobel) / 100
  total_for_sobel = 52.4 := by sorry

end percentage_for_sobel_l2098_209875


namespace g_composition_of_three_l2098_209847

def g (x : ℝ) : ℝ := 7 * x - 3

theorem g_composition_of_three : g (g (g 3)) = 858 := by
  sorry

end g_composition_of_three_l2098_209847


namespace jacoby_hourly_wage_l2098_209869

/-- Proves that Jacoby's hourly wage is $19 given the conditions of his savings and expenses --/
theorem jacoby_hourly_wage :
  let total_needed : ℕ := 5000
  let hours_worked : ℕ := 10
  let cookies_sold : ℕ := 24
  let cookie_price : ℕ := 4
  let lottery_win : ℕ := 500
  let sister_gift : ℕ := 500
  let remaining_needed : ℕ := 3214
  let hourly_wage : ℕ := (total_needed - remaining_needed - (cookies_sold * cookie_price) - lottery_win - 2 * sister_gift + 10) / hours_worked
  hourly_wage = 19
  := by sorry

end jacoby_hourly_wage_l2098_209869
