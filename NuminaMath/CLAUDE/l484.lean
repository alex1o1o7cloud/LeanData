import Mathlib

namespace fixed_point_of_function_l484_48440

theorem fixed_point_of_function (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (a - 1) * 2^x - 2*a
  f 1 = -2 := by sorry

end fixed_point_of_function_l484_48440


namespace percentage_of_lower_grades_with_cars_l484_48410

theorem percentage_of_lower_grades_with_cars
  (total_students : ℕ)
  (seniors : ℕ)
  (lower_grades : ℕ)
  (senior_car_percentage : ℚ)
  (total_car_percentage : ℚ)
  (h1 : total_students = 1200)
  (h2 : seniors = 300)
  (h3 : lower_grades = 900)
  (h4 : seniors + lower_grades = total_students)
  (h5 : senior_car_percentage = 1/2)
  (h6 : total_car_percentage = 1/5)
  : (total_car_percentage * total_students - senior_car_percentage * seniors) / lower_grades = 1/10 :=
by sorry

end percentage_of_lower_grades_with_cars_l484_48410


namespace tan_double_angle_special_case_l484_48444

theorem tan_double_angle_special_case (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end tan_double_angle_special_case_l484_48444


namespace cyclic_sum_inequality_l484_48498

theorem cyclic_sum_inequality (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 1 →
  (a^6 / ((a - b) * (a - c))) + (b^6 / ((b - c) * (b - a))) + (c^6 / ((c - a) * (c - b))) > 15 := by
  sorry

end cyclic_sum_inequality_l484_48498


namespace coupon_discount_percentage_l484_48467

theorem coupon_discount_percentage 
  (total_bill : ℝ) 
  (num_friends : ℕ) 
  (individual_payment : ℝ) 
  (h1 : total_bill = 100) 
  (h2 : num_friends = 5) 
  (h3 : individual_payment = 18.8) : 
  (total_bill - num_friends * individual_payment) / total_bill * 100 = 6 := by
sorry

end coupon_discount_percentage_l484_48467


namespace complex_equation_solution_l484_48434

theorem complex_equation_solution :
  ∃ z : ℂ, (z - Complex.I) * Complex.I = 2 + Complex.I ∧ z = 1 - Complex.I :=
by sorry

end complex_equation_solution_l484_48434


namespace valid_partition_exists_l484_48432

/-- Represents a person in the group -/
structure Person where
  id : Nat

/-- Represents the friendship and enmity relations in the group -/
structure Relations (P : Type) where
  friend : P → P
  enemy : P → P

/-- Represents a partition of the group into two subsets -/
structure Partition (P : Type) where
  set1 : Set P
  set2 : Set P
  partition_complete : set1 ∪ set2 = Set.univ
  partition_disjoint : set1 ∩ set2 = ∅

/-- The main theorem stating that a valid partition exists -/
theorem valid_partition_exists (P : Type) [Finite P] (r : Relations P) 
  (friend_injective : Function.Injective r.friend)
  (enemy_injective : Function.Injective r.enemy)
  (friend_enemy_distinct : ∀ p : P, r.friend p ≠ r.enemy p) :
  ∃ (part : Partition P), 
    (∀ p ∈ part.set1, r.friend p ∉ part.set1 ∧ r.enemy p ∉ part.set1) ∧
    (∀ p ∈ part.set2, r.friend p ∉ part.set2 ∧ r.enemy p ∉ part.set2) :=
  sorry

end valid_partition_exists_l484_48432


namespace linear_equation_equivalence_l484_48459

theorem linear_equation_equivalence (x y : ℝ) :
  (3 * x - 2 * y = 6) ↔ (y = (3 / 2) * x - 3) := by sorry

end linear_equation_equivalence_l484_48459


namespace coke_drinking_days_l484_48401

/-- Calculates the remaining days to finish drinking Coke -/
def remaining_days (total_volume : ℕ) (daily_consumption : ℕ) (days_consumed : ℕ) : ℕ :=
  (total_volume * 1000 / daily_consumption) - days_consumed

/-- Proves that it takes 7 more days to finish the Coke -/
theorem coke_drinking_days : remaining_days 2 200 3 = 7 := by
  sorry

end coke_drinking_days_l484_48401


namespace statement_1_statement_3_statement_4_l484_48415

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Define the planes and lines
variable (a b : Plane)
variable (l m n : Line)

-- Define the non-coincidence conditions
variable (planes_non_coincident : a ≠ b)
variable (lines_non_coincident : l ≠ m ∧ m ≠ n ∧ l ≠ n)

-- Theorem for statement 1
theorem statement_1 :
  parallel_planes a b →
  line_in_plane l a →
  parallel_line_plane l b :=
sorry

-- Theorem for statement 3
theorem statement_3 :
  parallel_line_plane l a →
  perpendicular_line_plane l b →
  perpendicular_planes a b :=
sorry

-- Theorem for statement 4
theorem statement_4 :
  skew_lines m n →
  parallel_line_plane m a →
  parallel_line_plane n a →
  perpendicular_lines l m →
  perpendicular_lines l n →
  perpendicular_line_plane l a :=
sorry

end statement_1_statement_3_statement_4_l484_48415


namespace win_sector_area_l484_48404

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 7) (h2 : p = 3/8) :
  p * π * r^2 = 147 * π / 8 := by
  sorry

end win_sector_area_l484_48404


namespace divides_a_iff_divides_n_l484_48427

/-- Sequence defined by a(n) = 2a(n-1) + a(n-2) for n > 1, with a(0) = 0 and a(1) = 1 -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

/-- For all natural numbers k and n, 2^k divides a(n) if and only if 2^k divides n -/
theorem divides_a_iff_divides_n (k n : ℕ) : (2^k : ℤ) ∣ a n ↔ 2^k ∣ n := by sorry

end divides_a_iff_divides_n_l484_48427


namespace jeff_fills_ten_boxes_l484_48472

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled (donuts_per_day : ℕ) (days : ℕ) (jeff_eats_per_day : ℕ) (chris_eats : ℕ) (donuts_per_box : ℕ) : ℕ :=
  ((donuts_per_day * days) - (jeff_eats_per_day * days) - chris_eats) / donuts_per_box

/-- Theorem stating that Jeff can fill 10 boxes with his donuts -/
theorem jeff_fills_ten_boxes :
  boxes_filled 10 12 1 8 10 = 10 := by
  sorry

end jeff_fills_ten_boxes_l484_48472


namespace division_result_l484_48408

theorem division_result : (0.08 : ℝ) / 0.002 = 40 := by sorry

end division_result_l484_48408


namespace no_formula_fits_all_pairs_l484_48480

-- Define the pairs of x and y values
def xy_pairs : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 35), (4, 69), (5, 119)]

-- Define the formulas
def formula_A (x : ℕ) : ℕ := x^3 + x^2 + x + 2
def formula_B (x : ℕ) : ℕ := 3*x^2 + 2*x + 1
def formula_C (x : ℕ) : ℕ := 2*x^3 - x + 4
def formula_D (x : ℕ) : ℕ := 3*x^3 + 2*x^2 + x + 1

-- Theorem statement
theorem no_formula_fits_all_pairs :
  ∀ (pair : ℕ × ℕ), pair ∈ xy_pairs →
    (formula_A pair.1 ≠ pair.2) ∧
    (formula_B pair.1 ≠ pair.2) ∧
    (formula_C pair.1 ≠ pair.2) ∧
    (formula_D pair.1 ≠ pair.2) :=
by sorry

end no_formula_fits_all_pairs_l484_48480


namespace sandys_book_purchase_l484_48481

theorem sandys_book_purchase (cost_shop1 : ℕ) (books_shop2 : ℕ) (cost_shop2 : ℕ) (avg_price : ℕ) : 
  cost_shop1 = 1480 →
  books_shop2 = 55 →
  cost_shop2 = 920 →
  avg_price = 20 →
  ∃ (books_shop1 : ℕ), 
    books_shop1 = 65 ∧ 
    (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = avg_price :=
by sorry

end sandys_book_purchase_l484_48481


namespace jasper_kite_raising_time_l484_48484

/-- Given Omar's kite-raising rate and Jasper's rate being three times Omar's,
    prove that Jasper takes 10 minutes to raise his kite 600 feet. -/
theorem jasper_kite_raising_time 
  (omar_height : ℝ) 
  (omar_time : ℝ) 
  (jasper_height : ℝ) 
  (omar_height_val : omar_height = 240) 
  (omar_time_val : omar_time = 12) 
  (jasper_height_val : jasper_height = 600) 
  (jasper_rate_mul : ℝ) 
  (jasper_rate_rel : jasper_rate_mul = 3) :
  (jasper_height / (jasper_rate_mul * omar_height / omar_time)) = 10 := by
  sorry

end jasper_kite_raising_time_l484_48484


namespace intersection_M_N_l484_48495

def M : Set ℤ := {-2, 0, 2}
def N : Set ℤ := {x | x^2 = x}

theorem intersection_M_N : M ∩ N = {0} := by sorry

end intersection_M_N_l484_48495


namespace fraction_to_decimal_l484_48414

theorem fraction_to_decimal : (5 : ℚ) / 8 = 0.625 := by
  sorry

end fraction_to_decimal_l484_48414


namespace camp_III_sample_size_l484_48455

def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ :=
  sorry

theorem camp_III_sample_size :
  systematic_sample 600 50 3 496 600 = 8 :=
sorry

end camp_III_sample_size_l484_48455


namespace coefficient_of_x_fifth_l484_48488

theorem coefficient_of_x_fifth (a : ℝ) : 
  (Nat.choose 8 5) * a^5 = 56 → a = 1 := by sorry

end coefficient_of_x_fifth_l484_48488


namespace closest_fraction_to_two_thirds_l484_48445

theorem closest_fraction_to_two_thirds :
  let fractions : List ℚ := [4/7, 9/14, 20/31, 61/95, 73/110]
  let target : ℚ := 2/3
  let differences := fractions.map (fun x => |x - target|)
  differences.minimum? = some |73/110 - 2/3| :=
by sorry

end closest_fraction_to_two_thirds_l484_48445


namespace rhombus_diagonals_l484_48483

/-- A rhombus with side length 1 and one angle of 120° has diagonals of length 1 and √3. -/
theorem rhombus_diagonals (s : ℝ) (α : ℝ) (d₁ d₂ : ℝ) 
  (h_side : s = 1)
  (h_angle : α = 120 * π / 180) :
  d₁ = 1 ∧ d₂ = Real.sqrt 3 := by
  sorry


end rhombus_diagonals_l484_48483


namespace marksman_hit_rate_l484_48423

theorem marksman_hit_rate (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →  -- p is a probability
  (1 - (1 - p)^4 = 80/81) →  -- probability of hitting at least once in 4 shots
  p = 2/3 := by
  sorry

end marksman_hit_rate_l484_48423


namespace treasure_chest_gems_l484_48431

theorem treasure_chest_gems (total_gems rubies : ℕ) 
  (h1 : total_gems = 5155)
  (h2 : rubies = 5110)
  (h3 : total_gems ≥ rubies) :
  total_gems - rubies = 45 := by
  sorry

end treasure_chest_gems_l484_48431


namespace janet_sock_purchase_l484_48418

theorem janet_sock_purchase : 
  ∀ (x y z : ℕ),
  -- Total number of pairs
  x + y + z = 18 →
  -- Total cost
  2*x + 5*y + 7*z = 60 →
  -- Exactly 3 pairs of $7 socks
  z = 3 →
  -- x represents the number of $2 socks
  x = 12 := by
sorry

end janet_sock_purchase_l484_48418


namespace cat_eye_movement_l484_48441

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the transformation (moving 3 units to the right)
def moveRight (p : Point) : Point :=
  (p.1 + 3, p.2)

-- Define the original points
def eye1 : Point := (-4, 3)
def eye2 : Point := (-2, 3)

-- State the theorem
theorem cat_eye_movement :
  (moveRight eye1 = (-1, 3)) ∧ (moveRight eye2 = (1, 3)) := by
  sorry

end cat_eye_movement_l484_48441


namespace remainder_theorem_example_l484_48465

theorem remainder_theorem_example (x : ℤ) :
  (Polynomial.X ^ 9 + 3 : Polynomial ℤ).eval 2 = 515 := by
  sorry

end remainder_theorem_example_l484_48465


namespace arrange_sticks_into_triangles_l484_48478

/-- Represents a stick with a positive length -/
structure Stick where
  length : ℝ
  positive : length > 0

/-- Represents a triangle formed by three sticks -/
structure Triangle where
  side1 : Stick
  side2 : Stick
  side3 : Stick

/-- Checks if three sticks can form a valid triangle -/
def isValidTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- Theorem stating that it's always possible to arrange six sticks into two triangles
    with one triangle having sides of one, two, and three sticks -/
theorem arrange_sticks_into_triangles
  (s1 s2 s3 s4 s5 s6 : Stick)
  (h_pairwise_different : s1.length < s2.length ∧ s2.length < s3.length ∧
                          s3.length < s4.length ∧ s4.length < s5.length ∧
                          s5.length < s6.length) :
  ∃ (t1 t2 : Triangle),
    (isValidTriangle t1.side1 t1.side2 t1.side3) ∧
    (isValidTriangle t2.side1 t2.side2 t2.side3) ∧
    ((t1.side1.length = s1.length ∧ t1.side2.length = s3.length + s5.length ∧ t1.side3.length = s2.length + s4.length + s6.length) ∨
     (t2.side1.length = s1.length ∧ t2.side2.length = s3.length + s5.length ∧ t2.side3.length = s2.length + s4.length + s6.length)) :=
by sorry

end arrange_sticks_into_triangles_l484_48478


namespace repeating_decimal_28_l484_48447

/-- The repeating decimal 0.2828... is equal to 28/99 -/
theorem repeating_decimal_28 : ∃ (x : ℚ), x = 28 / 99 ∧ x = 0 + (28 / 100) * (1 / (1 - 1 / 100)) :=
by
  sorry

end repeating_decimal_28_l484_48447


namespace probability_of_specific_draw_l484_48402

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 8

/-- The number of green balls in the bag -/
def green_balls : ℕ := 7

/-- The number of blue balls to be drawn -/
def blue_draw : ℕ := 3

/-- The number of green balls to be drawn -/
def green_draw : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := blue_balls + green_balls

/-- The total number of balls to be drawn -/
def total_draw : ℕ := blue_draw + green_draw

/-- The probability of drawing 3 blue balls followed by 2 green balls without replacement -/
theorem probability_of_specific_draw :
  (Nat.choose blue_balls blue_draw * Nat.choose green_balls green_draw : ℚ) /
  (Nat.choose total_balls total_draw : ℚ) = 1176 / 3003 := by
  sorry

end probability_of_specific_draw_l484_48402


namespace fat_per_cup_of_rice_l484_48436

/-- Amount of rice eaten in the morning -/
def morning_rice : ℕ := 3

/-- Amount of rice eaten in the afternoon -/
def afternoon_rice : ℕ := 2

/-- Amount of rice eaten in the evening -/
def evening_rice : ℕ := 5

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Total fat intake from rice in a week (in grams) -/
def weekly_fat_intake : ℕ := 700

/-- Calculate the amount of fat in a cup of rice -/
theorem fat_per_cup_of_rice : 
  (weekly_fat_intake : ℚ) / ((morning_rice + afternoon_rice + evening_rice) * days_in_week) = 10 := by
  sorry

end fat_per_cup_of_rice_l484_48436


namespace boat_current_rate_l484_48419

/-- Proves that the rate of the current is 5 km/hr given the conditions of the boat problem -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20) 
  (h2 : distance = 11.25) 
  (h3 : time_minutes = 27) : 
  ∃ current_rate : ℝ, 
    current_rate = 5 ∧ 
    distance = (boat_speed + current_rate) * (time_minutes / 60) :=
by
  sorry


end boat_current_rate_l484_48419


namespace man_son_age_ratio_l484_48477

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 27 years older than his son and the son's present age is 25 years. -/
theorem man_son_age_ratio :
  let son_age : ℕ := 25
  let man_age : ℕ := son_age + 27
  let son_age_in_two_years : ℕ := son_age + 2
  let man_age_in_two_years : ℕ := man_age + 2
  (man_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2 / 1 := by
  sorry

end man_son_age_ratio_l484_48477


namespace unique_n_for_consecutive_product_l484_48405

theorem unique_n_for_consecutive_product : ∃! (n : ℕ), 
  n > 0 ∧ ∃ (k : ℕ), k > 0 ∧ 
  (n^6 + 5*n^3 + 4*n + 116 = k * (k + 1) ∨ 
   n^6 + 5*n^3 + 4*n + 116 = k * (k + 1) * (k + 2) ∨
   n^6 + 5*n^3 + 4*n + 116 = k * (k + 1) * (k + 2) * (k + 3)) ∧
  n = 3 := by
sorry

end unique_n_for_consecutive_product_l484_48405


namespace arithmetic_progression_properties_l484_48413

/-- An arithmetic progression with given first and tenth terms -/
def ArithmeticProgression (a : ℕ → ℤ) : Prop :=
  a 1 = 21 ∧ a 10 = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_progression_properties (a : ℕ → ℤ) (h : ArithmeticProgression a) :
  (∀ n : ℕ, a n = -2 * n + 23) ∧
  (Finset.sum (Finset.range 11) a = 121) := by
  sorry

end arithmetic_progression_properties_l484_48413


namespace tom_family_members_l484_48476

/-- The number of family members Tom invited, excluding siblings -/
def family_members : ℕ := 2

/-- The number of Tom's siblings -/
def siblings : ℕ := 3

/-- The number of meals per day -/
def meals_per_day : ℕ := 3

/-- The number of plates used per meal -/
def plates_per_meal : ℕ := 2

/-- The duration of the stay in days -/
def stay_duration : ℕ := 4

/-- The total number of plates used -/
def total_plates : ℕ := 144

theorem tom_family_members :
  family_members = 2 :=
by sorry

end tom_family_members_l484_48476


namespace handshake_theorem_l484_48429

def num_people : ℕ := 8

def handshake_arrangements (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (n - 1) * handshake_arrangements (n - 2)

theorem handshake_theorem :
  handshake_arrangements num_people = 105 :=
by sorry

end handshake_theorem_l484_48429


namespace profit_division_l484_48461

theorem profit_division (profit_x profit_y total_profit : ℚ) : 
  profit_x / profit_y = 1/2 / (1/3) →
  profit_x - profit_y = 100 →
  profit_x + profit_y = total_profit →
  total_profit = 500 := by
sorry

end profit_division_l484_48461


namespace tutors_next_common_workday_l484_48421

def tim_schedule : ℕ := 5
def uma_schedule : ℕ := 6
def victor_schedule : ℕ := 9
def xavier_schedule : ℕ := 8

theorem tutors_next_common_workday : 
  lcm (lcm (lcm tim_schedule uma_schedule) victor_schedule) xavier_schedule = 360 := by
  sorry

end tutors_next_common_workday_l484_48421


namespace isosceles_triangle_locus_l484_48464

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def isIsosceles (t : Triangle) : Prop :=
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2

def satisfiesLocus (C : ℝ × ℝ) : Prop :=
  C.1^2 + C.2^2 - 6*C.1 + 4*C.2 - 5 = 0

theorem isosceles_triangle_locus :
  ∀ t : Triangle,
    t.A = (3, -2) →
    t.B = (0, 1) →
    isIsosceles t →
    t.C ≠ (0, 1) →
    t.C ≠ (6, -5) →
    satisfiesLocus t.C :=
  sorry

end isosceles_triangle_locus_l484_48464


namespace infinitely_many_expressible_l484_48492

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ k, a k < a (k + 1)

def expressible (a : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ (x y p q : ℕ), x > 0 ∧ y > 0 ∧ p ≠ q ∧ a m = x * a p + y * a q

theorem infinitely_many_expressible (a : ℕ → ℕ) 
  (h : is_strictly_increasing a) : 
  Set.Infinite {m : ℕ | expressible a m} :=
sorry

end infinitely_many_expressible_l484_48492


namespace sqrt_equation_solution_l484_48449

theorem sqrt_equation_solution (x y : ℝ) : 
  Real.sqrt (4 - 5*x + y) = 9 → y = 77 + 5*x := by
  sorry

end sqrt_equation_solution_l484_48449


namespace min_distance_to_i_l484_48446

-- Define the complex number z
variable (z : ℂ)

-- State the theorem
theorem min_distance_to_i (h : Complex.abs (z^2 - 1) = Complex.abs (z * (z - Complex.I))) :
  Complex.abs (z - Complex.I) ≥ (3 * Real.sqrt 2) / 4 := by
  sorry

end min_distance_to_i_l484_48446


namespace solutions_to_equation_unique_solutions_l484_48426

-- Define the equation
def equation (s : ℝ) : ℝ := 12 * s^2 + 2 * s

-- Theorem stating that 0.5 and -2/3 are solutions to the equation when t = 4
theorem solutions_to_equation :
  equation (1/2) = 4 ∧ equation (-2/3) = 4 :=
by sorry

-- Theorem stating that these are the only solutions
theorem unique_solutions (s : ℝ) :
  equation s = 4 ↔ s = 1/2 ∨ s = -2/3 :=
by sorry

end solutions_to_equation_unique_solutions_l484_48426


namespace eighteenth_digit_is_five_l484_48409

/-- The decimal expansion of 10000/9899 -/
def decimal_expansion : ℕ → ℕ
| 0 => 1  -- integer part
| 1 => 0  -- first decimal digit
| 2 => 1  -- second decimal digit
| n + 3 => (decimal_expansion (n + 1) + decimal_expansion (n + 2)) % 10

/-- The 18th digit after the decimal point in 10000/9899 is 5 -/
theorem eighteenth_digit_is_five : decimal_expansion 18 = 5 := by
  sorry

end eighteenth_digit_is_five_l484_48409


namespace pure_imaginary_condition_l484_48474

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ k : ℝ, m^2 + m - 2 + (m^2 - 1) * Complex.I = k * Complex.I) ↔ m = -2 := by
  sorry

end pure_imaginary_condition_l484_48474


namespace least_number_divisible_by_multiple_l484_48458

theorem least_number_divisible_by_multiple (n : ℕ) : n = 856 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 8) = 54 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 8) = 24 * k₁ ∧ (n + 8) = 32 * k₂ ∧ (n + 8) = 36 * k₃ ∧ (n + 8) = 54 * k₄) :=
by sorry

end least_number_divisible_by_multiple_l484_48458


namespace flower_count_l484_48430

theorem flower_count (red_green : ℕ) (red_yellow : ℕ) (green_yellow : ℕ)
  (h1 : red_green = 62)
  (h2 : red_yellow = 49)
  (h3 : green_yellow = 77) :
  ∃ (red green yellow : ℕ),
    red + green = red_green ∧
    red + yellow = red_yellow ∧
    green + yellow = green_yellow ∧
    red = 17 ∧ green = 45 ∧ yellow = 32 := by
  sorry

end flower_count_l484_48430


namespace vampire_consumption_l484_48424

/-- Represents the number of people consumed by the vampire and werewolf -/
structure Consumption where
  vampire : ℕ
  werewolf : ℕ

/-- The total consumption over a given number of weeks -/
def total_consumption (c : Consumption) (weeks : ℕ) : ℕ :=
  weeks * (c.vampire + c.werewolf)

theorem vampire_consumption (village_population : ℕ) (duration_weeks : ℕ) (c : Consumption) :
  village_population = 72 →
  duration_weeks = 9 →
  c.werewolf = 5 →
  total_consumption c duration_weeks = village_population →
  c.vampire = 3 := by
  sorry

end vampire_consumption_l484_48424


namespace log_problem_l484_48493

theorem log_problem (p q r x : ℝ) (d : ℝ) 
  (hp : Real.log x / Real.log p = 2)
  (hq : Real.log x / Real.log q = 3)
  (hr : Real.log x / Real.log r = 6)
  (hd : Real.log x / Real.log (p * q * r) = d)
  (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ x > 0) : d = 1 := by
  sorry

end log_problem_l484_48493


namespace cube_split_with_39_l484_48485

/-- Given a natural number m > 1, if m³ can be split into a sum of consecutive odd numbers 
    starting from (m+1)² and one of these odd numbers is 39, then m = 6 -/
theorem cube_split_with_39 (m : ℕ) (h1 : m > 1) :
  (∃ k : ℕ, (m + 1)^2 + 2*k = 39) → m = 6 := by
  sorry

end cube_split_with_39_l484_48485


namespace permutations_of_111222_l484_48466

/-- The number of permutations of a multiset with 6 elements, where 3 elements are of one type
    and 3 elements are of another type. -/
def permutations_of_multiset : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 3)

/-- The theorem states that the number of permutations of the multiset {1, 1, 1, 2, 2, 2}
    is equal to 20. -/
theorem permutations_of_111222 : permutations_of_multiset = 20 := by
  sorry

end permutations_of_111222_l484_48466


namespace existence_of_x_y_l484_48400

theorem existence_of_x_y : ∃ (x y : ℝ), 3*x + y > 0 ∧ 4*x + y > 0 ∧ 6*x + 5*y < 0 := by
  sorry

end existence_of_x_y_l484_48400


namespace increasing_geometric_sequence_formula_l484_48494

/-- An increasing geometric sequence with specific properties -/
def IncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (a 5)^2 = a 10 ∧
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))

/-- The general term formula for the sequence -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 2^n

/-- Theorem stating that an increasing geometric sequence with the given properties
    has the general term formula a_n = 2^n -/
theorem increasing_geometric_sequence_formula (a : ℕ → ℝ) :
  IncreasingGeometricSequence a → GeneralTermFormula a := by
  sorry

end increasing_geometric_sequence_formula_l484_48494


namespace arithmetic_sequence_12th_term_bound_l484_48460

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, its 12th term is less than or equal to 7. -/
theorem arithmetic_sequence_12th_term_bound
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_8 : a 8 ≥ 15)
  (h_9 : a 9 ≤ 13) :
  a 12 ≤ 7 :=
sorry

end arithmetic_sequence_12th_term_bound_l484_48460


namespace danny_bottle_caps_l484_48462

/-- The number of bottle caps Danny has after throwing some away and finding new ones -/
def final_bottle_caps (initial : ℕ) (thrown_away : ℕ) (found : ℕ) : ℕ :=
  initial - thrown_away + found

/-- Theorem stating that Danny's final bottle cap count is 67 -/
theorem danny_bottle_caps :
  final_bottle_caps 69 60 58 = 67 := by
  sorry

end danny_bottle_caps_l484_48462


namespace two_int_points_probability_l484_48453

/-- Square S with diagonal endpoints (1/2, 3/2) and (-1/2, -3/2) -/
def S : Set (ℝ × ℝ) := sorry

/-- Random point v = (x,y) where 0 ≤ x ≤ 1006 and 0 ≤ y ≤ 1006 -/
def v : ℝ × ℝ := sorry

/-- T(v) is a translated copy of S centered at v -/
def T (v : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The probability that T(v) contains exactly two integer points in its interior -/
def prob_two_int_points : ℝ := sorry

theorem two_int_points_probability :
  prob_two_int_points = 2 / 25 := by sorry

end two_int_points_probability_l484_48453


namespace motel_payment_savings_l484_48443

/-- Calculates the savings when choosing monthly payments over weekly payments for a motel stay. -/
theorem motel_payment_savings 
  (weeks_per_month : ℕ) 
  (total_months : ℕ) 
  (weekly_rate : ℕ) 
  (monthly_rate : ℕ) 
  (h1 : weeks_per_month = 4) 
  (h2 : total_months = 3) 
  (h3 : weekly_rate = 280) 
  (h4 : monthly_rate = 1000) : 
  (total_months * weeks_per_month * weekly_rate) - (total_months * monthly_rate) = 360 := by
  sorry

#check motel_payment_savings

end motel_payment_savings_l484_48443


namespace bells_lcm_l484_48422

/-- The time intervals at which the bells toll -/
def bell_intervals : List ℕ := [5, 8, 11, 15, 20]

/-- The theorem stating that the least common multiple of the bell intervals is 1320 -/
theorem bells_lcm : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 5 8) 11) 15) 20 = 1320 := by
  sorry

end bells_lcm_l484_48422


namespace question_ratio_l484_48450

/-- Represents the number of questions submitted by each person -/
structure QuestionSubmission where
  rajat : ℕ
  vikas : ℕ
  abhishek : ℕ

/-- The total number of questions submitted -/
def total_questions : ℕ := 24

/-- Theorem stating the ratio of questions submitted -/
theorem question_ratio (qs : QuestionSubmission) 
  (h1 : qs.rajat + qs.vikas + qs.abhishek = total_questions)
  (h2 : qs.vikas = 6) :
  ∃ (r a : ℕ), r = qs.rajat ∧ a = qs.abhishek ∧ r + a = 18 :=
by sorry

end question_ratio_l484_48450


namespace ninety_six_times_one_hundred_four_l484_48482

theorem ninety_six_times_one_hundred_four : 96 * 104 = 9984 := by
  sorry

end ninety_six_times_one_hundred_four_l484_48482


namespace domain_of_composition_l484_48499

-- Define the function f with domain [1,5]
def f : Set ℝ := Set.Icc 1 5

-- State the theorem
theorem domain_of_composition (f : Set ℝ) (h : f = Set.Icc 1 5) :
  {x : ℝ | ∃ y ∈ f, y = 2*x - 1} = Set.Icc 1 3 := by sorry

end domain_of_composition_l484_48499


namespace pirate_treasure_division_l484_48456

def pirate_share (n : ℕ) (k : ℕ) (remaining : ℚ) : ℚ :=
  (k : ℚ) / (n : ℚ) * remaining

def remaining_coins (n : ℕ) (k : ℕ) (initial : ℚ) : ℚ :=
  if k = 0 then initial
  else
    (1 - (k : ℚ) / (n : ℚ)) * remaining_coins n (k - 1) initial

def is_valid_distribution (n : ℕ) (initial : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → ∃ (m : ℕ), pirate_share n k (remaining_coins n (k - 1) initial) = m

theorem pirate_treasure_division (n : ℕ) (h : n = 15) :
  ∃ (initial : ℕ),
    (∀ smaller : ℕ, smaller < initial → ¬is_valid_distribution n smaller) ∧
    is_valid_distribution n initial ∧
    pirate_share n n (remaining_coins n (n - 1) initial) = 1536 := by
  sorry

end pirate_treasure_division_l484_48456


namespace fraction_zero_solution_l484_48471

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 + x - 2) / (x - 1) = 0 ∧ x ≠ 1 → x = -2 :=
by
  sorry

#check fraction_zero_solution

end fraction_zero_solution_l484_48471


namespace div_mul_calculation_l484_48470

theorem div_mul_calculation : (120 / 5) / 3 * 2 = 16 := by
  sorry

end div_mul_calculation_l484_48470


namespace equation_solutions_l484_48497

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ x₁ x₂ : ℝ, (2*x₁^2 - 6*x₁ = 3 ∧ 2*x₂^2 - 6*x₂ = 3) ∧ 
    x₁ = (3 + Real.sqrt 15) / 2 ∧ x₂ = (3 - Real.sqrt 15) / 2) :=
by sorry

end equation_solutions_l484_48497


namespace kyunghoon_descent_time_l484_48454

/-- Proves that given the conditions of Kyunghoon's mountain hike, the time it took him to go down is 2 hours. -/
theorem kyunghoon_descent_time :
  ∀ (d : ℝ), -- distance up the mountain
  d > 0 →
  d / 3 + (d + 2) / 4 = 4 → -- total time equation
  (d + 2) / 4 = 2 -- time to go down
  := by sorry

end kyunghoon_descent_time_l484_48454


namespace brad_profit_l484_48468

/-- Represents the sizes of lemonade glasses -/
inductive Size
| Small
| Medium
| Large

/-- Represents the lemonade stand data -/
structure LemonadeStand where
  yield_per_gallon : Size → ℕ
  cost_per_gallon : Size → ℚ
  price_per_glass : Size → ℚ
  gallons_made : Size → ℕ
  small_drunk : ℕ
  medium_bought : ℕ
  medium_spilled : ℕ
  large_unsold : ℕ

def brad_stand : LemonadeStand :=
  { yield_per_gallon := λ s => match s with
      | Size.Small => 16
      | Size.Medium => 10
      | Size.Large => 6
    cost_per_gallon := λ s => match s with
      | Size.Small => 2
      | Size.Medium => 7/2
      | Size.Large => 5
    price_per_glass := λ s => match s with
      | Size.Small => 1
      | Size.Medium => 7/4
      | Size.Large => 5/2
    gallons_made := λ _ => 2
    small_drunk := 4
    medium_bought := 3
    medium_spilled := 1
    large_unsold := 2 }

def total_cost (stand : LemonadeStand) : ℚ :=
  (stand.cost_per_gallon Size.Small * stand.gallons_made Size.Small) +
  (stand.cost_per_gallon Size.Medium * stand.gallons_made Size.Medium) +
  (stand.cost_per_gallon Size.Large * stand.gallons_made Size.Large)

def total_revenue (stand : LemonadeStand) : ℚ :=
  (stand.price_per_glass Size.Small * (stand.yield_per_gallon Size.Small * stand.gallons_made Size.Small - stand.small_drunk)) +
  (stand.price_per_glass Size.Medium * (stand.yield_per_gallon Size.Medium * stand.gallons_made Size.Medium - stand.medium_bought)) +
  (stand.price_per_glass Size.Large * (stand.yield_per_gallon Size.Large * stand.gallons_made Size.Large - stand.large_unsold))

def net_profit (stand : LemonadeStand) : ℚ :=
  total_revenue stand - total_cost stand

theorem brad_profit :
  net_profit brad_stand = 247/4 := by
  sorry

end brad_profit_l484_48468


namespace expression_equals_two_l484_48428

theorem expression_equals_two : (Real.sqrt 3)^2 + (4 - Real.pi)^0 - |(-3)| + Real.sqrt 2 * Real.cos (π / 4) = 2 := by
  sorry

end expression_equals_two_l484_48428


namespace investment_interest_rate_l484_48448

/-- Calculates the total interest rate for a two-share investment --/
def total_interest_rate (total_investment : ℚ) (rate1 : ℚ) (rate2 : ℚ) (amount2 : ℚ) : ℚ :=
  let amount1 := total_investment - amount2
  let interest1 := amount1 * rate1
  let interest2 := amount2 * rate2
  let total_interest := interest1 + interest2
  (total_interest / total_investment) * 100

/-- Theorem stating the total interest rate for the given investment scenario --/
theorem investment_interest_rate :
  total_interest_rate 10000 (9/100) (11/100) 3750 = (975/10000) * 100 := by
  sorry

end investment_interest_rate_l484_48448


namespace sphere_plane_intersection_l484_48457

/-- A sphere intersecting a plane creates a circular intersection. -/
theorem sphere_plane_intersection
  (r : ℝ) -- radius of the sphere
  (h : ℝ) -- depth of the intersection
  (w : ℝ) -- radius of the circular intersection
  (hr : r = 16.25)
  (hh : h = 10)
  (hw : w = 15) :
  r^2 = h * (2 * r - h) + w^2 :=
sorry

end sphere_plane_intersection_l484_48457


namespace correct_bucket_size_l484_48416

/-- The size of the bucket needed to collect leaking fluid -/
def bucket_size (leak_rate : ℝ) (max_time : ℝ) : ℝ :=
  2 * leak_rate * max_time

/-- Theorem stating the correct bucket size for the given conditions -/
theorem correct_bucket_size :
  bucket_size 1.5 12 = 36 := by
  sorry

end correct_bucket_size_l484_48416


namespace triangle_side_length_l484_48406

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (A + B + C = π) →
  -- Condition: √3 sin A + cos A = 2
  (Real.sqrt 3 * Real.sin A + Real.cos A = 2) →
  -- Condition: a = 3
  (a = 3) →
  -- Condition: C = 5π/12
  (C = 5 * π / 12) →
  -- Conclusion: b = √6
  (b = Real.sqrt 6) :=
by sorry

end triangle_side_length_l484_48406


namespace max_cross_section_area_l484_48433

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  base : List Point3D
  heights : List ℝ

def crossSectionArea (prism : TriangularPrism) (plane : Plane) : ℝ := sorry

/-- The main theorem statement -/
theorem max_cross_section_area :
  let prism : TriangularPrism := {
    base := [
      { x := 4, y := 0, z := 0 },
      { x := -2, y := 2 * Real.sqrt 3, z := 0 },
      { x := -2, y := -2 * Real.sqrt 3, z := 0 }
    ],
    heights := [2, 4, 3]
  }
  let plane : Plane := { a := 5, b := -3, c := 2, d := 30 }
  let area := crossSectionArea prism plane
  ∃ ε > 0, abs (area - 104.25) < ε := by
  sorry

end max_cross_section_area_l484_48433


namespace min_ratio_bounds_l484_48438

/-- An equiangular hexagon with alternating side lengths 1 and a -/
structure EquiangularHexagon :=
  (a : ℝ)

/-- A circle intersecting the hexagon at 12 distinct points -/
structure IntersectingCircle (h : EquiangularHexagon) :=
  (exists_intersection : True)

/-- The bounds M and N for the side length a -/
structure Bounds (h : EquiangularHexagon) (c : IntersectingCircle h) :=
  (M N : ℝ)
  (lower_bound : M < h.a)
  (upper_bound : h.a < N)

/-- The theorem stating the minimum possible value of N/M -/
theorem min_ratio_bounds 
  (h : EquiangularHexagon) 
  (c : IntersectingCircle h) 
  (b : Bounds h c) : 
  ∃ (M N : ℝ), M < h.a ∧ h.a < N ∧ 
  ∀ (M' N' : ℝ), M' < h.a → h.a < N' → (3 * Real.sqrt 3 + 3) / 2 ≤ N' / M' :=
sorry

end min_ratio_bounds_l484_48438


namespace equation_solution_l484_48451

theorem equation_solution : 
  ∀ x : ℤ, x * (x + 1) = 2014 * 2015 ↔ x = 2014 ∨ x = -2015 := by sorry

end equation_solution_l484_48451


namespace john_bought_36_rolls_l484_48487

/-- The number of rolls John bought given the cost per dozen and the amount spent -/
def rolls_bought (cost_per_dozen : ℚ) (amount_spent : ℚ) : ℚ :=
  (amount_spent / cost_per_dozen) * 12

/-- Theorem stating that John bought 36 rolls -/
theorem john_bought_36_rolls :
  let cost_per_dozen : ℚ := 5
  let amount_spent : ℚ := 15
  rolls_bought cost_per_dozen amount_spent = 36 := by
  sorry

end john_bought_36_rolls_l484_48487


namespace boat_distance_against_stream_l484_48489

/-- Calculates the distance a boat travels against the stream in one hour. -/
def distance_against_stream (boat_speed : ℝ) (distance_with_stream : ℝ) : ℝ :=
  boat_speed - (distance_with_stream - boat_speed)

/-- Theorem: Given a boat with speed 10 km/hr in still water that travels 15 km along the stream in one hour,
    the distance it travels against the stream in one hour is 5 km. -/
theorem boat_distance_against_stream :
  distance_against_stream 10 15 = 5 := by
  sorry

end boat_distance_against_stream_l484_48489


namespace valid_arrangements_count_l484_48437

/-- Represents the number of communities --/
def num_communities : ℕ := 3

/-- Represents the number of students --/
def num_students : ℕ := 4

/-- Represents the total number of arrangements without restrictions --/
def total_arrangements : ℕ := (num_students.choose 2) * (num_communities.factorial)

/-- Represents the number of arrangements where two specific students are in the same community --/
def same_community_arrangements : ℕ := num_communities.factorial

/-- The main theorem stating the number of valid arrangements --/
theorem valid_arrangements_count : 
  total_arrangements - same_community_arrangements = 30 :=
sorry

end valid_arrangements_count_l484_48437


namespace vershoks_in_arshin_l484_48420

/-- The number of vershoks in one arshin -/
def vershoks_per_arshin : ℕ := sorry

/-- Length of a plank in arshins -/
def plank_length : ℕ := 6

/-- Width of a plank in vershoks -/
def plank_width : ℕ := 6

/-- Side length of the room in arshins -/
def room_side : ℕ := 12

/-- Number of planks needed to cover the floor -/
def num_planks : ℕ := 64

theorem vershoks_in_arshin : 
  vershoks_per_arshin = 16 :=
by
  sorry

end vershoks_in_arshin_l484_48420


namespace score_order_l484_48475

theorem score_order (a b c d : ℝ) 
  (h1 : b + d = a + c)
  (h2 : a + c > b + d)
  (h3 : d > b + c)
  (ha : a ≥ 0)
  (hb : b ≥ 0)
  (hc : c ≥ 0)
  (hd : d ≥ 0) :
  a > d ∧ d > b ∧ b > c :=
by sorry

end score_order_l484_48475


namespace triangle_abc_is_obtuse_l484_48425

theorem triangle_abc_is_obtuse (A B C : ℝ) (h1 : A = 2 * B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A > 90 := by
  sorry

end triangle_abc_is_obtuse_l484_48425


namespace all_numbers_on_diagonal_l484_48442

/-- Represents a 15x15 table with numbers 1 to 15 -/
def Table := Fin 15 → Fin 15 → Fin 15

/-- The property that each number appears exactly once in each row -/
def row_property (t : Table) : Prop :=
  ∀ i j₁ j₂, j₁ ≠ j₂ → t i j₁ ≠ t i j₂

/-- The property that each number appears exactly once in each column -/
def column_property (t : Table) : Prop :=
  ∀ i₁ i₂ j, i₁ ≠ i₂ → t i₁ j ≠ t i₂ j

/-- The property that symmetrically placed numbers are identical -/
def symmetry_property (t : Table) : Prop :=
  ∀ i j, t i j = t j i

/-- The main theorem stating that all numbers appear on the main diagonal -/
theorem all_numbers_on_diagonal (t : Table)
  (h_row : row_property t)
  (h_col : column_property t)
  (h_sym : symmetry_property t) :
  ∀ n : Fin 15, ∃ i : Fin 15, t i i = n :=
sorry

end all_numbers_on_diagonal_l484_48442


namespace total_maggots_is_twenty_l484_48439

/-- The number of maggots served in the first attempt -/
def first_attempt : ℕ := 10

/-- The number of maggots served in the second attempt -/
def second_attempt : ℕ := 10

/-- The total number of maggots served -/
def total_maggots : ℕ := first_attempt + second_attempt

theorem total_maggots_is_twenty : total_maggots = 20 := by
  sorry

end total_maggots_is_twenty_l484_48439


namespace regions_in_circle_l484_48452

/-- The number of regions created by radii and concentric circles within a larger circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem: 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (r : ℕ) (c : ℕ) 
    (h1 : r = 16) (h2 : c = 10) : 
    num_regions r c = 176 := by
  sorry

#eval num_regions 16 10

end regions_in_circle_l484_48452


namespace parabola_vertex_l484_48417

/-- The parabola defined by y = (x-1)^2 + 2 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = (x-1)^2 + 2 has coordinates (1, 2) -/
theorem parabola_vertex : 
  ∀ (x : ℝ), parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 :=
by sorry

end parabola_vertex_l484_48417


namespace symmetric_axis_of_shifted_function_l484_48407

/-- Given a function f(x) = √3 * sin(2x) - cos(2x), prove that when shifted right by π/3 units,
    one of its symmetric axes is given by the equation x = π/6 -/
theorem symmetric_axis_of_shifted_function :
  ∃ (f : ℝ → ℝ) (g : ℝ → ℝ),
    (∀ x, f x = Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)) ∧
    (∀ x, g x = f (x - π / 3)) ∧
    (∀ x, g x = g (π / 3 - x)) := by
  sorry

end symmetric_axis_of_shifted_function_l484_48407


namespace all_statements_true_l484_48491

def A : Set ℝ := {-1, 2, 3}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem all_statements_true :
  (A ∩ B ≠ A) ∧
  (A ∪ B ≠ B) ∧
  (3 ∉ {x : ℝ | x < -1 ∨ x ≥ 3}) ∧
  (A ∩ {x : ℝ | x < -1 ∨ x ≥ 3} ≠ ∅) := by
  sorry

end all_statements_true_l484_48491


namespace reeta_pencils_l484_48490

theorem reeta_pencils (reeta_pencils : ℕ) 
  (h1 : reeta_pencils + (2 * reeta_pencils + 4) = 64) : 
  reeta_pencils = 20 := by
  sorry

end reeta_pencils_l484_48490


namespace complex_equation_solutions_l484_48435

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^3 + 2*z^2 + z - 2) / (z^2 - 3*z + 2)
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, f z = 0 :=
by sorry

end complex_equation_solutions_l484_48435


namespace students_not_participating_l484_48473

-- Define the sets and their cardinalities
def totalStudents : ℕ := 45
def volleyballParticipants : ℕ := 12
def trackFieldParticipants : ℕ := 20
def bothParticipants : ℕ := 6

-- Define the theorem
theorem students_not_participating : 
  totalStudents - volleyballParticipants - trackFieldParticipants + bothParticipants = 19 :=
by sorry

end students_not_participating_l484_48473


namespace race_outcomes_l484_48411

/-- The number of participants in the race -/
def num_participants : Nat := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_podium_positions : Nat := 3

/-- Represents whether a participant can finish in a specific position -/
def can_finish (participant : Nat) (position : Nat) : Prop :=
  ¬(participant = num_participants ∧ position = num_podium_positions)

/-- The number of valid race outcomes -/
def num_valid_outcomes : Nat := 120

theorem race_outcomes :
  (∀ (p₁ p₂ p₃ : Nat), p₁ ≤ num_participants → p₂ ≤ num_participants → p₃ ≤ num_participants →
    p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ →
    can_finish p₁ 1 → can_finish p₂ 2 → can_finish p₃ 3 →
    ∃! (outcome : Nat), outcome = num_valid_outcomes) :=
by sorry

end race_outcomes_l484_48411


namespace time_to_get_ahead_l484_48486

/-- Proves that the time for a faster traveler to get 1/3 mile ahead of a slower traveler is 2 minutes -/
theorem time_to_get_ahead (man_speed woman_speed : ℝ) (catch_up_time : ℝ) : 
  man_speed = 5 →
  woman_speed = 15 →
  catch_up_time = 4 →
  (woman_speed - man_speed) * 2 / 60 = 1 / 3 :=
by
  sorry

#check time_to_get_ahead

end time_to_get_ahead_l484_48486


namespace theater_ticket_difference_l484_48412

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
  orchestra_tickets + balcony_tickets = 370 →
  12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
  balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end theater_ticket_difference_l484_48412


namespace vector_dot_product_properties_l484_48496

/-- Given vectors a and b in R², prove properties about their dot product. -/
theorem vector_dot_product_properties (α β : ℝ) (k : ℝ) 
  (h_k_pos : k > 0)
  (a : ℝ × ℝ := (Real.cos α, Real.sin α))
  (b : ℝ × ℝ := (Real.cos β, Real.sin β))
  (h_norm : ‖k • a + b‖ = Real.sqrt 3 * ‖a - k • b‖) :
  let dot := a.1 * b.1 + a.2 * b.2
  ∃ θ : ℝ,
    (dot = Real.cos (α - β)) ∧ 
    (dot = (k^2 + 1) / (4 * k)) ∧
    (0 ≤ θ ∧ θ ≤ π) ∧
    (dot ≥ 1/2) ∧
    (dot = 1/2 ↔ θ = π/3) :=
sorry

end vector_dot_product_properties_l484_48496


namespace smallest_coconut_pile_l484_48469

def process (n : ℕ) : ℕ := (n - 1) * 4 / 5

def iterate_process (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | m + 1 => process (iterate_process n m)

theorem smallest_coconut_pile :
  ∃ (n : ℕ), n > 0 ∧ 
    (iterate_process n 5) % 5 = 0 ∧
    n ≥ (iterate_process n 0) - (iterate_process n 1) +
        (iterate_process n 1) - (iterate_process n 2) +
        (iterate_process n 2) - (iterate_process n 3) +
        (iterate_process n 3) - (iterate_process n 4) +
        (iterate_process n 4) - (iterate_process n 5) + 5 ∧
    (∀ (m : ℕ), m > 0 ∧ m < n →
      (iterate_process m 5) % 5 ≠ 0 ∨
      m < (iterate_process m 0) - (iterate_process m 1) +
          (iterate_process m 1) - (iterate_process m 2) +
          (iterate_process m 2) - (iterate_process m 3) +
          (iterate_process m 3) - (iterate_process m 4) +
          (iterate_process m 4) - (iterate_process m 5) + 5) ∧
    n = 3121 := by
  sorry

#check smallest_coconut_pile

end smallest_coconut_pile_l484_48469


namespace function_monotonicity_implies_c_zero_and_b_positive_l484_48463

-- Define the function f(x)
def f (b c x : ℝ) : ℝ := -x^3 - b*x^2 - 5*c*x

-- State the theorem
theorem function_monotonicity_implies_c_zero_and_b_positive
  (b c : ℝ)
  (h1 : ∀ x ≤ 0, Monotone (fun x => f b c x))
  (h2 : ∀ x ∈ Set.Icc 0 6, StrictMono (fun x => f b c x)) :
  c = 0 ∧ b > 0 := by
  sorry

end function_monotonicity_implies_c_zero_and_b_positive_l484_48463


namespace content_paths_count_l484_48479

/-- Represents the grid structure of the "CONTENT" word pattern --/
def ContentGrid : Type := Unit  -- Placeholder for the grid structure

/-- Represents a valid path in the ContentGrid --/
def ValidPath (grid : ContentGrid) : Type := Unit  -- Placeholder for path representation

/-- Counts the number of valid paths in the ContentGrid --/
def countValidPaths (grid : ContentGrid) : ℕ := sorry

/-- The main theorem stating that the number of valid paths is 127 --/
theorem content_paths_count (grid : ContentGrid) : countValidPaths grid = 127 := by
  sorry

end content_paths_count_l484_48479


namespace min_value_product_l484_48403

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1 / 2) :
  (x + y) * (2 * y + 3 * z) * (x * z + 2) ≥ 4 * Real.sqrt 6 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' * y' * z' = 1 / 2 ∧
    (x' + y') * (2 * y' + 3 * z') * (x' * z' + 2) = 4 * Real.sqrt 6 := by
  sorry

end min_value_product_l484_48403
