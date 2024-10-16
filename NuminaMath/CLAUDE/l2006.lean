import Mathlib

namespace NUMINAMATH_CALUDE_lcm_problem_l2006_200606

theorem lcm_problem (n : ℕ+) (h1 : Nat.lcm 40 n = 200) (h2 : Nat.lcm n 45 = 180) : n = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2006_200606


namespace NUMINAMATH_CALUDE_fib_linear_combination_fib_quadratic_combination_l2006_200645

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Part (a)
theorem fib_linear_combination (a b : ℝ) :
  (∀ n : ℕ, ∃ k : ℕ, a * fib n + b * fib (n + 1) = fib k) ↔
  ∃ k : ℕ, a = fib (k - 1) ∧ b = fib k :=
sorry

-- Part (b)
theorem fib_quadratic_combination (u v : ℝ) :
  (u > 0 ∧ v > 0 ∧ ∀ n : ℕ, ∃ k : ℕ, u * (fib n)^2 + v * (fib (n + 1))^2 = fib k) ↔
  u = 1 ∧ v = 1 :=
sorry

end NUMINAMATH_CALUDE_fib_linear_combination_fib_quadratic_combination_l2006_200645


namespace NUMINAMATH_CALUDE_expression_evaluation_l2006_200641

theorem expression_evaluation (a b c d : ℝ) :
  d = c + 5 →
  c = b - 8 →
  b = a + 3 →
  a = 3 →
  a - 1 ≠ 0 →
  d - 6 ≠ 0 →
  c + 4 ≠ 0 →
  ((a + 3) / (a - 1)) * ((d - 3) / (d - 6)) * ((c + 9) / (c + 4)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2006_200641


namespace NUMINAMATH_CALUDE_main_theorem_l2006_200643

/-- The set S of ordered triples satisfying the given conditions -/
def S (n : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {t | ∃ x y z, t = (x, y, z) ∧ 
       x ∈ Finset.range n ∧ y ∈ Finset.range n ∧ z ∈ Finset.range n ∧
       ((x < y ∧ y < z) ∨ (y < z ∧ z < x) ∨ (z < x ∧ x < y)) ∧
       ¬((x < y ∧ y < z) ∧ (y < z ∧ z < x)) ∧
       ¬((y < z ∧ z < x) ∧ (z < x ∧ x < y)) ∧
       ¬((z < x ∧ x < y) ∧ (x < y ∧ y < z))}

/-- The main theorem -/
theorem main_theorem (n : ℕ) (h : n ≥ 4) 
  (x y z w : ℕ) (hxyz : (x, y, z) ∈ S n) (hzwx : (z, w, x) ∈ S n) :
  (y, z, w) ∈ S n ∧ (x, y, w) ∈ S n := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l2006_200643


namespace NUMINAMATH_CALUDE_c_rent_share_is_72_l2006_200650

/-- Represents the rent share calculation for a pasture -/
def RentShare (total_rent : ℚ) (oxen_a : ℕ) (months_a : ℕ) (oxen_b : ℕ) (months_b : ℕ) (oxen_c : ℕ) (months_c : ℕ) : ℚ :=
  let total_oxen_months := oxen_a * months_a + oxen_b * months_b + oxen_c * months_c
  let c_oxen_months := oxen_c * months_c
  (c_oxen_months : ℚ) / total_oxen_months * total_rent

/-- Theorem stating that C's share of the rent is approximately 72 -/
theorem c_rent_share_is_72 :
  let rent_share := RentShare 280 10 7 12 5 15 3
  ∃ ε > 0, abs (rent_share - 72) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_c_rent_share_is_72_l2006_200650


namespace NUMINAMATH_CALUDE_square_of_1007_l2006_200651

theorem square_of_1007 : (1007 : ℕ) ^ 2 = 1014049 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1007_l2006_200651


namespace NUMINAMATH_CALUDE_solution_to_equation_l2006_200611

theorem solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (6 * x) ^ 5 = (12 * x) ^ 4 ∧ x = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2006_200611


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_max_distance_line_equation_l2006_200665

-- Define the line l
def line_l (a b x y : ℝ) : Prop := (2*a + b)*x + (a + b)*y + a - b = 0

-- Define the fixed point A
def point_A : ℝ × ℝ := (-2, 3)

-- Define point P
def point_P : ℝ × ℝ := (3, 4)

-- Theorem 1: Line l always passes through point A
theorem line_passes_through_fixed_point (a b : ℝ) :
  line_l a b (point_A.1) (point_A.2) := by sorry

-- Theorem 2: The line passing through A that maximizes distance to P has equation 5x + y + 7 = 0
theorem max_distance_line_equation :
  ∃ (k m : ℝ), 
    (k * point_A.1 + point_A.2 + m = 0) ∧ 
    (k * point_P.1 + point_P.2 + m = 0) ∧ 
    (k = 5 ∧ m = 7) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_max_distance_line_equation_l2006_200665


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l2006_200695

theorem maximize_x_cubed_y_fourth :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 27 →
  x^3 * y^4 ≤ (81/7)^3 * (108/7)^4 :=
by sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l2006_200695


namespace NUMINAMATH_CALUDE_back_parking_spaces_l2006_200696

/-- Proves that the number of parking spaces in the back of the school is 38 -/
theorem back_parking_spaces : 
  ∀ (front_spaces back_spaces : ℕ),
    front_spaces = 52 →
    ∃ (parked_cars available_spaces : ℕ),
      parked_cars = 39 ∧
      available_spaces = 32 ∧
      parked_cars + available_spaces = front_spaces + back_spaces ∧
      parked_cars - front_spaces = back_spaces / 2 →
        back_spaces = 38 := by
sorry

end NUMINAMATH_CALUDE_back_parking_spaces_l2006_200696


namespace NUMINAMATH_CALUDE_minimum_opponents_l2006_200659

/-- 
Given two integers h ≥ 1 and p ≥ 2, this theorem states that the minimum number of 
pairs of opponents in an hp-member parliament, such that in every partition into h 
houses of p members each, some house contains at least one pair of opponents, 
is equal to min((h-1)p + 1, (h+1 choose 2)).
-/
theorem minimum_opponents (h p : ℕ) (h_ge_one : h ≥ 1) (p_ge_two : p ≥ 2) :
  let parliament_size := h * p
  let min_opponents := min ((h - 1) * p + 1) (Nat.choose (h + 1) 2)
  ∀ (opponents : Finset (Finset (Fin parliament_size))),
    (∀ partition : Finset (Finset (Fin parliament_size)),
      (partition.card = h ∧ 
       ∀ house ∈ partition, house.card = p ∧
       partition.sup id = Finset.univ) →
      ∃ house ∈ partition, ∃ pair ∈ opponents, pair ⊆ house) →
    opponents.card ≥ min_opponents ∧
    ∃ opponents_min : Finset (Finset (Fin parliament_size)),
      opponents_min.card = min_opponents ∧
      (∀ partition : Finset (Finset (Fin parliament_size)),
        (partition.card = h ∧ 
         ∀ house ∈ partition, house.card = p ∧
         partition.sup id = Finset.univ) →
        ∃ house ∈ partition, ∃ pair ∈ opponents_min, pair ⊆ house) :=
by sorry

end NUMINAMATH_CALUDE_minimum_opponents_l2006_200659


namespace NUMINAMATH_CALUDE_constant_function_shift_l2006_200685

/-- Given a function f that is constant 2 for all real numbers, 
    prove that f(x + 2) = 2 for all real numbers x. -/
theorem constant_function_shift (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 2) :
  ∀ x : ℝ, f (x + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_constant_function_shift_l2006_200685


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2006_200683

-- Define the arithmetic sequence
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := sorry

-- State the theorem
theorem arithmetic_geometric_ratio (d : ℝ) :
  d ≠ 0 →
  (∃ r : ℝ, arithmetic_sequence d 3 = arithmetic_sequence d 1 * r ∧ 
            arithmetic_sequence d 4 = arithmetic_sequence d 3 * r) →
  (arithmetic_sequence d 1 + arithmetic_sequence d 5 + arithmetic_sequence d 17) / 
  (arithmetic_sequence d 2 + arithmetic_sequence d 6 + arithmetic_sequence d 18) = 8 / 11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2006_200683


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l2006_200681

def n : ℕ := 2014

theorem rearrangement_theorem (x y : Fin n → ℤ)
  (hx : ∀ i j, i ≠ j → x i % n ≠ x j % n)
  (hy : ∀ i j, i ≠ j → y i % n ≠ y j % n) :
  ∃ σ : Equiv.Perm (Fin n), ∀ i j, i ≠ j → (x i + y (σ i)) % (2 * n) ≠ (x j + y (σ j)) % (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l2006_200681


namespace NUMINAMATH_CALUDE_apple_tree_problem_l2006_200636

/-- The number of apples Rachel picked from the tree -/
def apples_picked : ℝ := 7.5

/-- The number of new apples that grew on the tree after Rachel picked -/
def new_apples : ℝ := 2.3

/-- The number of apples currently on the tree -/
def current_apples : ℝ := 6.2

/-- The original number of apples on the tree -/
def original_apples : ℝ := apples_picked + current_apples - new_apples

theorem apple_tree_problem :
  original_apples = 11.4 := by sorry

end NUMINAMATH_CALUDE_apple_tree_problem_l2006_200636


namespace NUMINAMATH_CALUDE_workers_payment_schedule_l2006_200639

theorem workers_payment_schedule (total_days : ℕ) (pay_per_day_worked : ℤ) (pay_returned_per_day_not_worked : ℤ) 
  (h1 : total_days = 30)
  (h2 : pay_per_day_worked = 100)
  (h3 : pay_returned_per_day_not_worked = 25)
  (h4 : ∃ (days_worked days_not_worked : ℕ), 
    days_worked + days_not_worked = total_days ∧ 
    pay_per_day_worked * days_worked - pay_returned_per_day_not_worked * days_not_worked = 0) :
  ∃ (days_not_worked : ℕ), days_not_worked = 24 := by
sorry

end NUMINAMATH_CALUDE_workers_payment_schedule_l2006_200639


namespace NUMINAMATH_CALUDE_grandfather_money_calculation_l2006_200680

def birthday_money_problem (aunt_money grandfather_money total_money bank_money : ℕ) : Prop :=
  aunt_money = 75 ∧
  bank_money = 45 ∧
  bank_money = total_money / 5 ∧
  total_money = aunt_money + grandfather_money

theorem grandfather_money_calculation :
  ∀ aunt_money grandfather_money total_money bank_money,
  birthday_money_problem aunt_money grandfather_money total_money bank_money →
  grandfather_money = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_grandfather_money_calculation_l2006_200680


namespace NUMINAMATH_CALUDE_star_seven_three_l2006_200667

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := 4*a + 3*b - a*b

-- State the theorem
theorem star_seven_three : star 7 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_three_l2006_200667


namespace NUMINAMATH_CALUDE_jessica_cut_forty_roses_l2006_200655

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_vase : ℕ) (final_vase : ℕ) (returned_to_sarah : ℕ) (total_garden : ℕ) : ℕ :=
  (final_vase - initial_vase) + returned_to_sarah

/-- Theorem stating that Jessica cut 40 roses from her garden -/
theorem jessica_cut_forty_roses :
  roses_cut 7 37 10 84 = 40 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_forty_roses_l2006_200655


namespace NUMINAMATH_CALUDE_total_tickets_sold_l2006_200613

/-- Represents the price of an adult ticket in dollars -/
def adult_price : ℝ := 4

/-- Represents the price of a student ticket in dollars -/
def student_price : ℝ := 2.5

/-- Represents the total revenue from ticket sales in dollars -/
def total_revenue : ℝ := 222.5

/-- Represents the number of student tickets sold -/
def student_tickets : ℕ := 9

/-- Theorem stating the total number of tickets sold -/
theorem total_tickets_sold : ℕ := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l2006_200613


namespace NUMINAMATH_CALUDE_system_solution_l2006_200662

theorem system_solution : 
  ∃ (x y z : ℝ), 
    (x + y + z = 15 ∧ 
     x^2 + y^2 + z^2 = 81 ∧ 
     x*y + x*z = 3*y*z) ∧
    ((x = 6 ∧ y = 3 ∧ z = 6) ∨ 
     (x = 6 ∧ y = 6 ∧ z = 3)) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l2006_200662


namespace NUMINAMATH_CALUDE_triangle_area_l2006_200658

/-- Triangle XYZ with given properties has area 35√7/2 -/
theorem triangle_area (X Y Z : Real) (r R : Real) (h1 : r = 3) (h2 : R = 12) 
  (h3 : 3 * Real.cos Y = Real.cos X + Real.cos Z) : 
  ∃ (area : Real), area = (35 * Real.sqrt 7) / 2 ∧ 
  area = r * (Real.sin X * R + Real.sin Y * R + Real.sin Z * R) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2006_200658


namespace NUMINAMATH_CALUDE_special_function_bound_l2006_200649

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x * y) = f x * f y) ∧
  (∀ x y, f (x + y) ≤ 2 * (f x + f y)) ∧
  (f 2 = 4)

/-- For any function satisfying the SpecialFunction properties, f(3) ≤ 9 -/
theorem special_function_bound (f : ℝ → ℝ) (hf : SpecialFunction f) : f 3 ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_bound_l2006_200649


namespace NUMINAMATH_CALUDE_polygon_sequence_limit_l2006_200656

/-- Represents the sequence of polygons formed by cutting corners -/
def polygon_sequence (n : ℕ) : ℝ :=
  sorry

/-- The area of the triangle cut from each corner in the nth iteration -/
def cut_triangle_area (n : ℕ) : ℝ :=
  sorry

/-- The number of corners in the nth polygon -/
def num_corners (n : ℕ) : ℕ :=
  sorry

theorem polygon_sequence_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |polygon_sequence n - 5/7| < ε :=
sorry

end NUMINAMATH_CALUDE_polygon_sequence_limit_l2006_200656


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2006_200624

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 3/2]
def interval : Set ℝ := Set.Icc (-3) (3/2)

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -18 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2006_200624


namespace NUMINAMATH_CALUDE_ellipse_properties_l2006_200617

/-- Ellipse C passing through points A(2,0) and B(0,1) -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Line on which point P lies -/
def line_P (x y : ℝ) : Prop := x + y = 4

/-- Point Q on ellipse C -/
def point_Q (x y : ℝ) : Prop := ellipse_C x y

/-- Parallelogram condition for PAQB -/
def is_parallelogram (px py qx qy : ℝ) : Prop :=
  px + qx = 2 ∧ py + qy = 1

theorem ellipse_properties :
  ∃ (e : ℝ),
    (∀ x y, ellipse_C x y ↔ x^2 / 4 + y^2 = 1) ∧
    e = Real.sqrt 3 / 2 ∧
    ∃ px py qx qy,
      line_P px py ∧
      point_Q qx qy ∧
      is_parallelogram px py qx qy ∧
      ((px = 18/5 ∧ py = 2/5) ∨ (px = 2 ∧ py = 2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2006_200617


namespace NUMINAMATH_CALUDE_xy_minus_ten_squared_ge_64_l2006_200646

theorem xy_minus_ten_squared_ge_64 (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 := by
  sorry

end NUMINAMATH_CALUDE_xy_minus_ten_squared_ge_64_l2006_200646


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2006_200654

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def num_O : ℕ := 5

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := num_N * atomic_weight_N + num_O * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 108.02 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2006_200654


namespace NUMINAMATH_CALUDE_min_value_system_l2006_200653

theorem min_value_system (x y k : ℝ) :
  (3 * x + y ≥ 0) →
  (4 * x + 3 * y ≥ k) →
  (∀ x' y', (3 * x' + y' ≥ 0) → (4 * x' + 3 * y' ≥ k) → (2 * x' + 4 * y' ≥ 2 * x + 4 * y)) →
  (2 * x + 4 * y = -6) →
  (k ≤ 0 ∧ ∀ m : ℤ, m > 0 → ¬(k ≥ m)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_system_l2006_200653


namespace NUMINAMATH_CALUDE_balls_without_holes_l2006_200618

theorem balls_without_holes 
  (total_soccer : ℕ) 
  (total_basketball : ℕ) 
  (soccer_with_holes : ℕ) 
  (basketball_with_holes : ℕ) 
  (h1 : total_soccer = 40) 
  (h2 : total_basketball = 15) 
  (h3 : soccer_with_holes = 30) 
  (h4 : basketball_with_holes = 7) : 
  (total_soccer - soccer_with_holes) + (total_basketball - basketball_with_holes) = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_balls_without_holes_l2006_200618


namespace NUMINAMATH_CALUDE_fourteenSidedFigure_area_l2006_200616

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
def Polygon := List Point

/-- The fourteen-sided figure described in the problem -/
def fourteenSidedFigure : Polygon := [
  ⟨1, 2⟩, ⟨2, 3⟩, ⟨4, 3⟩, ⟨5, 4⟩, ⟨5, 6⟩, ⟨6, 7⟩, ⟨7, 6⟩, ⟨7, 4⟩,
  ⟨6, 3⟩, ⟨4, 3⟩, ⟨3, 2⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨1, 2⟩
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ :=
  sorry -- Actual implementation would go here

/-- Theorem: The area of the fourteen-sided figure is 14 cm² -/
theorem fourteenSidedFigure_area :
  calculateArea fourteenSidedFigure = 14 := by
  sorry -- Proof would go here

end NUMINAMATH_CALUDE_fourteenSidedFigure_area_l2006_200616


namespace NUMINAMATH_CALUDE_prob_adjacent_knights_l2006_200648

/-- The number of knights at the round table -/
def n : ℕ := 30

/-- The number of knights chosen for the quest -/
def k : ℕ := 4

/-- The probability of choosing k knights from n such that at least two are adjacent -/
def prob_adjacent (n k : ℕ) : ℚ :=
  1 - (n * (n - 3) * (n - 4) * (n - 5) : ℚ) / (n.choose k : ℚ)

/-- The theorem stating the probability of choosing 4 knights from 30 such that at least two are adjacent -/
theorem prob_adjacent_knights : prob_adjacent n k = 53 / 183 := by sorry

end NUMINAMATH_CALUDE_prob_adjacent_knights_l2006_200648


namespace NUMINAMATH_CALUDE_yellow_ball_packs_l2006_200640

theorem yellow_ball_packs (red_packs green_packs balls_per_pack total_balls : ℕ) 
  (h1 : red_packs = 3)
  (h2 : green_packs = 8)
  (h3 : balls_per_pack = 19)
  (h4 : total_balls = 399) :
  ∃ yellow_packs : ℕ, 
    yellow_packs * balls_per_pack + red_packs * balls_per_pack + green_packs * balls_per_pack = total_balls ∧
    yellow_packs = 10 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_packs_l2006_200640


namespace NUMINAMATH_CALUDE_popcorn_probability_l2006_200668

theorem popcorn_probability (total : ℝ) (h_total_pos : 0 < total) : 
  let white := (3/4 : ℝ) * total
  let yellow := (1/4 : ℝ) * total
  let white_popped := (3/5 : ℝ) * white
  let yellow_popped := (3/4 : ℝ) * yellow
  let total_popped := white_popped + yellow_popped
  (white_popped / total_popped) = (12/17 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_popcorn_probability_l2006_200668


namespace NUMINAMATH_CALUDE_green_light_probability_theorem_l2006_200657

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of arriving during the green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (d.red + d.yellow + d.green)

/-- Theorem stating the probability of arriving during the green light
    for the given traffic light durations -/
theorem green_light_probability_theorem (d : TrafficLightDuration) 
  (h1 : d.red = 30)
  (h2 : d.yellow = 5)
  (h3 : d.green = 40) :
  greenLightProbability d = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_green_light_probability_theorem_l2006_200657


namespace NUMINAMATH_CALUDE_song_listens_theorem_l2006_200634

def calculate_total_listens (initial_listens : ℕ) (months : ℕ) : ℕ :=
  let doubling_factor := 2 ^ months
  initial_listens * (doubling_factor - 1) + initial_listens

theorem song_listens_theorem (initial_listens : ℕ) (months : ℕ) 
  (h1 : initial_listens = 60000) (h2 : months = 3) :
  calculate_total_listens initial_listens months = 900000 := by
  sorry

#eval calculate_total_listens 60000 3

end NUMINAMATH_CALUDE_song_listens_theorem_l2006_200634


namespace NUMINAMATH_CALUDE_total_saltwater_animals_l2006_200621

theorem total_saltwater_animals (num_aquariums : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : num_aquariums = 26)
  (h2 : animals_per_aquarium = 2) :
  num_aquariums * animals_per_aquarium = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_saltwater_animals_l2006_200621


namespace NUMINAMATH_CALUDE_line_perp_plane_condition_l2006_200625

-- Define the types for lines and planes
variable (L P : Type) [NormedAddCommGroup L] [NormedSpace ℝ L] [NormedAddCommGroup P] [NormedSpace ℝ P]

-- Define the perpendicular relation
variable (perpendicular : L → L → Prop)
variable (perpendicular_plane : L → P → Prop)

-- Define the subset relation
variable (subset : L → P → Prop)

-- Theorem statement
theorem line_perp_plane_condition (l m : L) (α : P) 
  (h_subset : subset m α) :
  (∀ l m α, perpendicular_plane l α → perpendicular l m) ∧ 
  (∃ l m α, perpendicular l m ∧ ¬perpendicular_plane l α) :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_condition_l2006_200625


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2006_200694

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I + 2) :
  z.im = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2006_200694


namespace NUMINAMATH_CALUDE_valid_arrangements_l2006_200601

/-- Represents the number of ways to arrange 7 people in a line -/
def arrangement_count : ℕ := 72

/-- Represents the total number of people -/
def total_people : ℕ := 7

/-- Represents the number of students -/
def student_count : ℕ := 6

/-- Represents whether two people are at the ends of the line -/
def are_at_ends (coach : ℕ) (student_a : ℕ) : Prop :=
  (coach = 1 ∧ student_a = total_people) ∨ (coach = total_people ∧ student_a = 1)

/-- Represents whether two students are adjacent in the line -/
def are_adjacent (student1 : ℕ) (student2 : ℕ) : Prop :=
  student1 + 1 = student2 ∨ student2 + 1 = student1

/-- Represents whether two students are not adjacent in the line -/
def are_not_adjacent (student1 : ℕ) (student2 : ℕ) : Prop :=
  ¬(are_adjacent student1 student2)

/-- Theorem stating that the number of valid arrangements is 72 -/
theorem valid_arrangements :
  ∀ (coach student_a student_b student_c student_d : ℕ),
    are_at_ends coach student_a →
    are_adjacent student_b student_c →
    are_not_adjacent student_b student_d →
    arrangement_count = 72 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_l2006_200601


namespace NUMINAMATH_CALUDE_catherine_pens_problem_l2006_200622

theorem catherine_pens_problem (initial_pens initial_pencils : ℕ) :
  initial_pens = initial_pencils →
  initial_pens - 7 * 8 + initial_pencils - 7 * 6 = 22 →
  initial_pens = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_catherine_pens_problem_l2006_200622


namespace NUMINAMATH_CALUDE_abs_z_eq_one_l2006_200669

theorem abs_z_eq_one (z : ℂ) (h : (1 - Complex.I) / z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_eq_one_l2006_200669


namespace NUMINAMATH_CALUDE_fermat_like_theorem_l2006_200676

theorem fermat_like_theorem (x y z n : ℕ) (h : n ≥ z) : x^n + y^n ≠ z^n := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_theorem_l2006_200676


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l2006_200666

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  r^2 = 13/4 → 
  π * r^2 / 2 = 13*π/8 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l2006_200666


namespace NUMINAMATH_CALUDE_range_of_a_l2006_200637

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) ∧ 
  (∃ x : ℝ, x^2 + 4*x + a = 0) → 
  a ∈ Set.Icc (Real.exp 1) 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2006_200637


namespace NUMINAMATH_CALUDE_max_value_rational_function_max_value_attained_l2006_200600

theorem max_value_rational_function (x : ℝ) :
  x^6 / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) ≤ 1/37 :=
by sorry

theorem max_value_attained :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) = 1/37 :=
by sorry

end NUMINAMATH_CALUDE_max_value_rational_function_max_value_attained_l2006_200600


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2006_200678

theorem triangle_angle_measure (A B C : Real) (a b c : Real) (S : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  -- A = 2B
  (A = 2 * B) →
  -- Area S = a²/4
  (S = a^2 / 4) →
  -- Area formula
  (S = (1/2) * b * c * Real.sin A) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusion: A is either π/2 or π/4
  (A = π/2 ∨ A = π/4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2006_200678


namespace NUMINAMATH_CALUDE_tangent_line_a_value_l2006_200619

/-- A line in polar coordinates tangent to a circle. -/
structure PolarLineTangentToCircle where
  a : ℝ
  tangent_line : ℝ → ℝ → Prop
  circle : ℝ → ℝ → Prop
  a_positive : a > 0
  is_tangent : ∀ θ ρ, tangent_line ρ θ ↔ ρ * (Real.cos θ + Real.sin θ) = a
  circle_eq : ∀ θ ρ, circle ρ θ ↔ ρ = 2 * Real.cos θ

/-- The value of 'a' for a line tangent to the given circle is 1 + √2. -/
theorem tangent_line_a_value (h : PolarLineTangentToCircle) : h.a = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_a_value_l2006_200619


namespace NUMINAMATH_CALUDE_dave_had_18_tickets_l2006_200693

/-- Calculates the number of tickets Dave had left after playing games and receiving tickets from a friend -/
def daves_tickets : ℕ :=
  let first_set := 14 - 2
  let second_set := 8 - 5
  let third_set := (first_set * 3) - 15
  let total_after_games := first_set + second_set + third_set
  let after_buying_toys := total_after_games - 25
  after_buying_toys + 7

/-- Theorem stating that Dave had 18 tickets left -/
theorem dave_had_18_tickets : daves_tickets = 18 := by
  sorry

end NUMINAMATH_CALUDE_dave_had_18_tickets_l2006_200693


namespace NUMINAMATH_CALUDE_student_hostel_cost_theorem_l2006_200607

/-- The cost per day for additional weeks in a student youth hostel -/
def additional_week_cost (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  let first_week_cost := 7 * first_week_daily_rate
  let additional_days := total_days - 7
  let additional_cost := total_cost - first_week_cost
  additional_cost / additional_days

theorem student_hostel_cost_theorem (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) 
  (h1 : first_week_daily_rate = 18)
  (h2 : total_days = 23)
  (h3 : total_cost = 334) :
  additional_week_cost first_week_daily_rate total_days total_cost = 13 := by
  sorry

end NUMINAMATH_CALUDE_student_hostel_cost_theorem_l2006_200607


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l2006_200684

def workday_hours : ℕ := 8
def minutes_per_hour : ℕ := 60
def first_meeting_duration : ℕ := 40
def third_meeting_duration : ℕ := 30
def overlap_duration : ℕ := 10

def total_workday_minutes : ℕ := workday_hours * minutes_per_hour

def second_meeting_duration : ℕ := 2 * first_meeting_duration

def effective_second_meeting_duration : ℕ := second_meeting_duration - overlap_duration

def total_meeting_time : ℕ := first_meeting_duration + effective_second_meeting_duration + third_meeting_duration

def meeting_percentage : ℚ := (total_meeting_time : ℚ) / (total_workday_minutes : ℚ) * 100

theorem workday_meeting_percentage :
  ∃ (x : ℚ), abs (meeting_percentage - x) < 1 ∧ ⌊x⌋ = 29 := by sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l2006_200684


namespace NUMINAMATH_CALUDE_max_factors_of_b_power_n_l2006_200603

def is_prime (p : ℕ) : Prop := sorry

-- Function to count factors of a number
def count_factors (n : ℕ) : ℕ := sorry

-- Function to check if a number is the product of exactly two distinct primes less than 15
def is_product_of_two_primes_less_than_15 (b : ℕ) : Prop := sorry

theorem max_factors_of_b_power_n :
  ∃ (b n : ℕ),
    b ≤ 15 ∧
    n ≤ 15 ∧
    is_product_of_two_primes_less_than_15 b ∧
    count_factors (b^n) = 256 ∧
    ∀ (b' n' : ℕ),
      b' ≤ 15 →
      n' ≤ 15 →
      is_product_of_two_primes_less_than_15 b' →
      count_factors (b'^n') ≤ 256 := by
  sorry

end NUMINAMATH_CALUDE_max_factors_of_b_power_n_l2006_200603


namespace NUMINAMATH_CALUDE_marbles_shared_proof_l2006_200628

/-- The number of marbles Carolyn started with -/
def initial_marbles : ℕ := 47

/-- The number of marbles Carolyn ended up with after sharing -/
def final_marbles : ℕ := 5

/-- The number of marbles Carolyn shared -/
def shared_marbles : ℕ := initial_marbles - final_marbles

theorem marbles_shared_proof : shared_marbles = 42 := by
  sorry

end NUMINAMATH_CALUDE_marbles_shared_proof_l2006_200628


namespace NUMINAMATH_CALUDE_cookies_per_person_l2006_200690

theorem cookies_per_person (batches : ℕ) (dozen_per_batch : ℕ) (people : ℕ) :
  batches = 4 →
  dozen_per_batch = 2 →
  people = 16 →
  (batches * dozen_per_batch * 12) / people = 6 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_person_l2006_200690


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2006_200672

-- Define the inequality
def inequality (k x : ℝ) : Prop :=
  k * (x^2 + 6*x - k) * (x^2 + x - 12) > 0

-- Define the solution set
def solution_set (k : ℝ) : Set ℝ :=
  {x | inequality k x}

-- Theorem statement
theorem inequality_solution_set (k : ℝ) :
  solution_set k = Set.Ioo (-4 : ℝ) 3 ↔ k ∈ Set.Iic (-9 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2006_200672


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l2006_200691

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 12 * x - 4 * y + 20

-- Define the square
structure Square where
  side_length : ℝ
  parallel_to_x_axis : Prop

-- Define the inscribed circle
structure InscribedCircle where
  equation : (ℝ → ℝ → Prop)
  square : Square

-- Theorem statement
theorem inscribed_circle_square_area 
  (circle : InscribedCircle) 
  (h : circle.equation = circle_equation) :
  circle.square.side_length^2 = 80 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l2006_200691


namespace NUMINAMATH_CALUDE_gift_price_gift_price_exact_l2006_200686

/-- The price of Lisa's gift given her savings and contributions from family and friends --/
theorem gift_price (lisa_savings : ℚ) (mother_fraction : ℚ) (brother_multiplier : ℚ) 
  (friend_fraction : ℚ) (short_amount : ℚ) : ℚ :=
  let mother_contribution := mother_fraction * lisa_savings
  let brother_contribution := brother_multiplier * mother_contribution
  let friend_contribution := friend_fraction * (mother_contribution + brother_contribution)
  let total_contributions := lisa_savings + mother_contribution + brother_contribution + friend_contribution
  total_contributions + short_amount

/-- The price of Lisa's gift is $3935.71 --/
theorem gift_price_exact : 
  gift_price 1600 (3/8) (5/4) (2/7) 600 = 3935.71 := by
  sorry

end NUMINAMATH_CALUDE_gift_price_gift_price_exact_l2006_200686


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l2006_200670

theorem relationship_between_exponents 
  (a b c d : ℝ) (x y q z : ℝ) 
  (h1 : a^(x+1) = c^(q+2)) 
  (h2 : a^(x+1) = b) 
  (h3 : c^(y+3) = a^(z+4)) 
  (h4 : c^(y+3) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  (q+2)*(z+4) = (y+3)*(x+1) := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l2006_200670


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2006_200642

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ (M ∩ P) → x ∈ (M ∪ P)) ∧
  (∃ x, x ∈ (M ∪ P) ∧ x ∉ (M ∩ P)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2006_200642


namespace NUMINAMATH_CALUDE_two_zeros_iff_m_in_range_l2006_200644

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (2 * log x - x) + 1 / x^2 - 1 / x

theorem two_zeros_iff_m_in_range (m : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧
    ∀ z : ℝ, 0 < z → f m z = 0 → (z = x ∨ z = y)) ↔
  m ∈ Set.Ioo (1 / (8 * (log 2 - 1))) 0 :=
sorry

end NUMINAMATH_CALUDE_two_zeros_iff_m_in_range_l2006_200644


namespace NUMINAMATH_CALUDE_complex_number_intersection_l2006_200664

theorem complex_number_intersection (M N : Set ℂ) (i : ℂ) (z : ℂ) : 
  M = {1, 2, z*i} → 
  N = {3, 4} → 
  M ∩ N = {4} → 
  i^2 = -1 →
  z = -4*i := by sorry

end NUMINAMATH_CALUDE_complex_number_intersection_l2006_200664


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2006_200675

theorem retail_price_calculation (W S P R : ℚ) : 
  W = 99 → 
  S = 0.9 * P → 
  R = 0.2 * W → 
  S = W + R → 
  P = 132 := by
sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l2006_200675


namespace NUMINAMATH_CALUDE_b_months_is_nine_l2006_200671

/-- Represents the pasture rental scenario -/
structure PastureRental where
  total_cost : ℝ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℝ

/-- Theorem stating that given the conditions, b put in horses for 9 months -/
theorem b_months_is_nine (pr : PastureRental)
  (h1 : pr.total_cost = 435)
  (h2 : pr.a_horses = 12)
  (h3 : pr.a_months = 8)
  (h4 : pr.b_horses = 16)
  (h5 : pr.c_horses = 18)
  (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 180) :
  ∃ x : ℝ, x = 9 ∧ 
    pr.b_payment = (pr.total_cost / (pr.a_horses * pr.a_months + pr.b_horses * x + pr.c_horses * pr.c_months)) * (pr.b_horses * x) :=
by sorry


end NUMINAMATH_CALUDE_b_months_is_nine_l2006_200671


namespace NUMINAMATH_CALUDE_quadratic_not_always_positive_l2006_200663

theorem quadratic_not_always_positive : ¬ (∀ x : ℝ, x^2 + 3*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_not_always_positive_l2006_200663


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2006_200660

/-- An isosceles triangle with perimeter 24 and a median that divides the perimeter in a 5:3 ratio -/
structure IsoscelesTriangle where
  /-- Length of each equal side -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The perimeter is 24 -/
  perimeter_eq : 2 * x + y = 24
  /-- The median divides the perimeter in a 5:3 ratio -/
  median_ratio : 3 * x / (x + y) = 5 / 3

/-- The base of the isosceles triangle is 4 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2006_200660


namespace NUMINAMATH_CALUDE_coconut_flavored_jelly_beans_l2006_200632

theorem coconut_flavored_jelly_beans (total : ℕ) (red_fraction : ℚ) (coconut_fraction : ℚ) :
  total = 4000 →
  red_fraction = 3 / 4 →
  coconut_fraction = 1 / 4 →
  (total * red_fraction * coconut_fraction : ℚ) = 750 := by
  sorry

end NUMINAMATH_CALUDE_coconut_flavored_jelly_beans_l2006_200632


namespace NUMINAMATH_CALUDE_triangle_inequality_l2006_200629

/-- A triangle with sides x, y, and z satisfies the inequality
    (x+y+z)(x+y-z)(x+z-y)(z+y-x) ≤ 4x²y² -/
theorem triangle_inequality (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y > z ∧ x + z > y ∧ y + z > x) :
  (x + y + z) * (x + y - z) * (x + z - y) * (z + y - x) ≤ 4 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2006_200629


namespace NUMINAMATH_CALUDE_grain_oil_production_growth_l2006_200610

theorem grain_oil_production_growth (x : ℝ) : 
  (450000 * (1 + x)^2 = 500000) ↔ 
  (∃ (y : ℝ), 450000 * (1 + x) = y ∧ y * (1 + x) = 500000) :=
sorry

end NUMINAMATH_CALUDE_grain_oil_production_growth_l2006_200610


namespace NUMINAMATH_CALUDE_number_problem_l2006_200604

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) : 
  (40/100) * N = 120 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2006_200604


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l2006_200679

theorem triangle_side_calculation (A B C : Real) (a b : Real) : 
  B = π / 6 → -- 30° in radians
  C = 7 * π / 12 → -- 105° in radians
  A = π / 4 → -- 45° in radians (derived from B + C + A = π)
  a = 4 →
  b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l2006_200679


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l2006_200652

theorem sin_cos_equation_solution :
  ∃ x : ℝ, x = π / 14 ∧ Real.sin (3 * x) * Real.sin (4 * x) = Real.cos (3 * x) * Real.cos (4 * x) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l2006_200652


namespace NUMINAMATH_CALUDE_final_position_is_37_steps_behind_l2006_200612

/-- Represents the walking challenge rules -/
def walkingChallenge (n : ℕ) : ℤ :=
  if n = 1 then 0
  else if Nat.Prime n then 2
  else -3

/-- The final position after completing all 30 moves -/
def finalPosition : ℤ :=
  -(Finset.sum (Finset.range 30) (fun i => walkingChallenge (i + 1)))

/-- Theorem stating the final position is 37 steps behind the starting point -/
theorem final_position_is_37_steps_behind :
  finalPosition = -37 := by sorry

end NUMINAMATH_CALUDE_final_position_is_37_steps_behind_l2006_200612


namespace NUMINAMATH_CALUDE_fraction_equality_implies_k_l2006_200682

theorem fraction_equality_implies_k (x y z k : ℝ) :
  (9 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 15 / (z - y)) →
  k = 24 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_k_l2006_200682


namespace NUMINAMATH_CALUDE_average_weight_increase_l2006_200699

theorem average_weight_increase (original_group_size : ℕ) 
  (original_weight : ℝ) (new_weight : ℝ) : 
  original_group_size = 5 → 
  original_weight = 50 → 
  new_weight = 70 → 
  (new_weight - original_weight) / original_group_size = 4 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2006_200699


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l2006_200602

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l2006_200602


namespace NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l2006_200638

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x

theorem f_monotonicity_and_m_range :
  ∀ (a b : ℝ),
  (∀ (x : ℝ), x > 0 → f a b x = a * Real.log x - b * x) →
  (b = 1 →
    ((a ≤ 0 → ∀ (x y : ℝ), 0 < x → x < y → f a b y < f a b x) ∧
     (a > 0 → (∀ (x y : ℝ), 0 < x → x < y → y < a → f a b x < f a b y) ∧
              (∀ (x y : ℝ), a < x → x < y → f a b y < f a b x)))) ∧
  (a = 1 →
    (∀ (x : ℝ), x > 0 → f a b x ≤ -1) →
    (∀ (x : ℝ), x > 0 → f a b x ≤ x * Real.exp x - (b + 1) * x - 1) →
    b ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l2006_200638


namespace NUMINAMATH_CALUDE_infinitely_many_mtrp_numbers_l2006_200626

/-- Sum of digits in decimal representation of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Definition of MTRP-number -/
def is_mtrp_number (m n : ℕ) : Prop :=
  n > 0 ∧ n % m = 1 ∧ sum_of_digits (n^2) ≥ sum_of_digits n

theorem infinitely_many_mtrp_numbers (m : ℕ) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_mtrp_number m n := by sorry

end NUMINAMATH_CALUDE_infinitely_many_mtrp_numbers_l2006_200626


namespace NUMINAMATH_CALUDE_expression_value_l2006_200698

theorem expression_value : 
  Real.sqrt (2018 * 2021 * 2022 * 2023 + 2024^2) - 2024^2 = -12138 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2006_200698


namespace NUMINAMATH_CALUDE_train_crossing_time_l2006_200661

/-- Proves that a train of given length and speed takes a specific time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2006_200661


namespace NUMINAMATH_CALUDE_candle_flower_groupings_l2006_200647

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem candle_flower_groupings :
  (choose 6 3) * (choose 15 12) = 9100 := by
sorry

end NUMINAMATH_CALUDE_candle_flower_groupings_l2006_200647


namespace NUMINAMATH_CALUDE_average_weight_increase_problem_solution_l2006_200620

/-- The increase in average weight when replacing a person in a group -/
theorem average_weight_increase (n : ℕ) (old_weight new_weight : ℝ) : 
  n > 0 → (new_weight - old_weight) / n = (new_weight - old_weight) / n := by
  sorry

/-- The specific case of the problem -/
theorem problem_solution : 
  let n : ℕ := 10
  let old_weight : ℝ := 65
  let new_weight : ℝ := 137
  (new_weight - old_weight) / n = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_problem_solution_l2006_200620


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2006_200615

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 2)^2 + 5) →
  QuadraticFunction a b c 1 = 2 →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2006_200615


namespace NUMINAMATH_CALUDE_sally_reading_time_l2006_200689

/-- The number of pages Sally reads on a weekday -/
def weekday_pages : ℕ := 10

/-- The number of pages Sally reads on a weekend day -/
def weekend_pages : ℕ := 20

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of pages in Sally's book -/
def book_pages : ℕ := 180

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- Theorem stating that it takes Sally 2 weeks to finish the book -/
theorem sally_reading_time :
  weekday_pages * weekdays + weekend_pages * weekend_days = book_pages / weeks_to_finish :=
by sorry

end NUMINAMATH_CALUDE_sally_reading_time_l2006_200689


namespace NUMINAMATH_CALUDE_rectangle_area_l2006_200688

theorem rectangle_area (a b : ℕ) : 
  (2 * (a + b) = 16) →
  (a^2 + b^2 - 2*a*b - 4 = 0) →
  (a * b = 15) := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2006_200688


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2006_200614

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 60 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2006_200614


namespace NUMINAMATH_CALUDE_abs_two_implies_two_or_neg_two_l2006_200631

theorem abs_two_implies_two_or_neg_two (x : ℝ) : |x| = 2 → x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_implies_two_or_neg_two_l2006_200631


namespace NUMINAMATH_CALUDE_exists_cousin_180_problems_l2006_200623

/-- Represents the homework scenario for me and my cousin -/
structure HomeworkScenario where
  p : ℕ+  -- My rate (problems per hour)
  t : ℕ+  -- Time I take to finish homework (hours)
  n : ℕ   -- Number of problems I complete

/-- Calculates the number of problems my cousin does -/
def cousin_problems (s : HomeworkScenario) : ℕ :=
  ((3 * s.p.val - 5) * (s.t.val + 3)) / 2

/-- Theorem stating that there exists a scenario where my cousin does 180 problems -/
theorem exists_cousin_180_problems :
  ∃ (s : HomeworkScenario), 
    s.p ≥ 15 ∧ 
    s.n = s.p.val * s.t.val ∧ 
    cousin_problems s = 180 := by
  sorry

end NUMINAMATH_CALUDE_exists_cousin_180_problems_l2006_200623


namespace NUMINAMATH_CALUDE_rhombus_area_from_intersecting_strips_l2006_200627

/-- The area of a rhombus formed by two intersecting strips -/
theorem rhombus_area_from_intersecting_strips (α : ℝ) (h_α : 0 < α ∧ α < π) :
  let strip_width : ℝ := 1
  let rhombus_side : ℝ := strip_width / Real.sin α
  let rhombus_area : ℝ := rhombus_side * strip_width
  rhombus_area = 1 / Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_from_intersecting_strips_l2006_200627


namespace NUMINAMATH_CALUDE_sqrt_equation_implies_power_l2006_200635

theorem sqrt_equation_implies_power (x y : ℝ) : 
  Real.sqrt (2 - x) + Real.sqrt (x - 2) + y = 4 → x^y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_implies_power_l2006_200635


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_three_l2006_200674

theorem sum_of_roots_equals_three : ∃ (P Q : ℝ), P + Q = 3 ∧ 3 * P^2 - 9 * P + 6 = 0 ∧ 3 * Q^2 - 9 * Q + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_three_l2006_200674


namespace NUMINAMATH_CALUDE_hannah_son_cutting_rate_l2006_200692

/-- The number of strands Hannah's son can cut per minute -/
def sonCuttingRate (totalStrands : ℕ) (hannahRate : ℕ) (totalTime : ℕ) : ℕ :=
  (totalStrands - hannahRate * totalTime) / totalTime

theorem hannah_son_cutting_rate :
  sonCuttingRate 22 8 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hannah_son_cutting_rate_l2006_200692


namespace NUMINAMATH_CALUDE_trapezoid_height_l2006_200687

theorem trapezoid_height (upper_side lower_side area height : ℝ) : 
  upper_side = 5 →
  lower_side = 9 →
  area = 56 →
  area = (1/2) * (upper_side + lower_side) * height →
  height = 8 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_height_l2006_200687


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l2006_200605

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l2006_200605


namespace NUMINAMATH_CALUDE_negation_of_all_integers_squared_geq_one_l2006_200609

theorem negation_of_all_integers_squared_geq_one :
  (¬ ∀ (x : ℤ), x^2 ≥ 1) ↔ (∃ (x : ℤ), x^2 < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_integers_squared_geq_one_l2006_200609


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2006_200677

-- Define the given parameters
def train_length : ℝ := 140
def bridge_length : ℝ := 235
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmhr := speed_ms * conversion_factor
  speed_kmhr = 45 := by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2006_200677


namespace NUMINAMATH_CALUDE_system_solution_l2006_200630

theorem system_solution (x y z : ℕ) : 
  x + y + z = 12 → 
  4 * x + 3 * y + 2 * z = 36 → 
  x ∈ ({0, 1, 2, 3, 4, 5, 6} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2006_200630


namespace NUMINAMATH_CALUDE_work_completion_time_l2006_200673

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 10

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days A leaves before the work is completed -/
def a_leave_before : ℝ := 5

/-- The total number of days to complete the work -/
def total_days : ℝ := 10

/-- Theorem stating that given the conditions, the work is completed in 10 days -/
theorem work_completion_time :
  (1 / a_days + 1 / b_days) * (total_days - a_leave_before) + (1 / b_days) * a_leave_before = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2006_200673


namespace NUMINAMATH_CALUDE_negation_of_implication_l2006_200697

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2006_200697


namespace NUMINAMATH_CALUDE_fifth_number_21st_row_is_809_l2006_200633

/-- The number of odd numbers in the nth row of the pattern -/
def oddNumbersInRow (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers in the first n rows -/
def sumOddNumbersInRows (n : ℕ) : ℕ :=
  (oddNumbersInRow n + 1) * n / 2

/-- The nth positive odd number -/
def nthPositiveOdd (n : ℕ) : ℕ := 2 * n - 1

theorem fifth_number_21st_row_is_809 :
  let totalPreviousRows := sumOddNumbersInRows 20
  let positionInSequence := totalPreviousRows + 5
  nthPositiveOdd positionInSequence = 809 := by sorry

end NUMINAMATH_CALUDE_fifth_number_21st_row_is_809_l2006_200633


namespace NUMINAMATH_CALUDE_sheela_deposit_l2006_200608

/-- Calculates the deposit amount given a monthly income and deposit percentage -/
def deposit_amount (monthly_income : ℕ) (deposit_percentage : ℚ) : ℚ :=
  (deposit_percentage * monthly_income : ℚ)

theorem sheela_deposit :
  deposit_amount 10000 (25 / 100) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_sheela_deposit_l2006_200608
